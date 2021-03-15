from ogb.nodeproppred import PygNodePropPredDataset
from kungfu.cmd import launch_multiprocess
import argparse
import multiprocessing as mp
import os
import os.path as osp
import time

import queue

import torch
import torch.nn.functional as F
from torch.distributed import rpc

from quiver.cuda_sampler import CudaNeighborSampler
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
from tqdm import tqdm


class SyncConfig:
    def __init__(self, rank, world_size, queues, barriers, timeout, reduce,
                 dist):
        self.rank = rank
        self.world_size = world_size
        self.queues = queues
        self.barriers = barriers
        self.timeout = timeout
        self.reduce = reduce
        self.dist = dist


local_rank = None
loader = None


def sample_cuda(nodes, size):
    torch.cuda.set_device(local_rank)
    nodes = loader.global2local(nodes)
    neighbors, counts = loader.quiver.sample_neighbor(0, nodes, size)
    neighbors = loader.local2global(neighbors)

    return neighbors, counts


def sample_cpu(nodes, size):
    nodes = loader.global2local(nodes)
    neighbors, counts = loader.quiver.sample_neighbor(nodes, size)
    neighbors = loader.local2global(neighbors)

    return neighbors, counts


def node_f(nodes, is_feature):
    cpu = torch.device('cpu')
    if is_feature:
        return loader.x[nodes].to(cpu)
    else:
        return loader.y[nodes].to(cpu)


def SamplerProcess(sync, dev, edge_index, data, train_idx, sizes, batch_size):
    global loader
    global local_rank
    device, num = dev
    group = sync.rank // sync.world_size
    sync.rank = sync.rank % sync.world_size

    def node2rank(nodes):
        ranks = torch.fmod(nodes, sync.world_size)
        return ranks

    def local2global(nodes):
        return nodes

    def global2local(nodes):
        return nodes

    if device != 'cpu':
        torch.cuda.set_device(device)
        if sync.dist:
            import quiver.dist_cuda_sampler as dist
            comm = dist.Comm(sync.rank, sync.world_size)
            local_rank = sync.rank % 4
            loader = dist.SyncDistNeighborSampler(
                comm, (int(edge_index.max() + 1), edge_index, data,
                       local2global, global2local, node2rank),
                train_idx, [15, 10, 5],
                device,
                node_f,
                batch_size=batch_size)
            dist.sample_neighbor = sample_cuda
        else:
            loader = CudaNeighborSampler(edge_index,
                                         device=device,
                                         rank=num,
                                         node_idx=train_idx,
                                         sizes=sizes,
                                         batch_size=batch_size,
                                         shuffle=True)
    else:
        if sync.dist:
            import quiver.dist_cpu_sampler as dist
            comm = dist.Comm(sync.rank, sync.world_size)
            loader = dist.SyncDistNeighborSampler(
                comm, (int(edge_index.max() + 1), edge_index,
                       torch.zeros(1, dtype=torch.long), local2global,
                       global2local, node2rank),
                train_idx, [15, 10, 5],
                sync.rank,
                batch_size=batch_size)
            dist.sample_neighbor = sample_cpu
        else:
            loader = NeighborSampler(edge_index,
                                     node_idx=train_idx,
                                     sizes=sizes,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num)
    if sync.dist:
        os.environ['MASTER_ADDR'] = 'localhost'
        if device == 'cpu':
            group += 32
        port = str(29500 + 100 * group)
        os.environ['MASTER_PORT'] = port
        rpc.init_rpc(f"worker{sync.rank}",
                     rank=sync.rank,
                     world_size=sync.world_size,
                     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                         num_worker_threads=1, rpc_timeout=20))
    cont = True
    count = 0
    wait = 0.0
    while cont:
        for sample in loader:
            count += 1
            if count == 1:
                sync.barriers[0].wait()
                print('beg sample')
            try:
                t0 = time.time()
                sync.queues[0].put((device, sample), timeout=sync.timeout)
                wait += time.time() - t0
            except queue.Full:
                cont = False
                break
    print(dev)
    print('sample wait: ' + str(wait))


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)


def TrainProcess(rank, sync, num_batch, num_features, num_hidden, num_classes,
                 num_layers):
    if sync.reduce:
        import kungfu.torch as kf
        from kungfu.python import current_cluster_size, current_rank
    device = torch.device('cuda:' +
                          str(rank) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(rank)
    model = SAGE(num_features, num_hidden, num_classes, num_layers)
    model = model.to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    sync.rank = rank
    if sync.reduce:
        optimizer = kf.optimizers.SynchronousSGDOptimizer(
            optimizer, named_parameters=model.named_parameters())
        kf.broadcast_parameters(model.state_dict())

    model.train()
    cpu_count = gpu_count = 0
    wait = 0.0
    wait2 = 0.0
    qs = []
    rd = 0.0
    s = torch.cuda.Stream()
    sync.barriers[0].wait()
    for i in range(num_batch):
        if i % 10 == 9:
            qs.append(sync.queues[0].qsize())
        try:
            t0 = time.time()
            dev, sample = sync.queues[0].get(timeout=sync.timeout)
            w = time.time() - t0
            wait += w
        except queue.Empty:
            print('wait too long')
            continue
        if dev == 'cpu':
            cpu_count += 1
        else:
            gpu_count += 1
        feature, label, adjs = sample
        adjs = [adj.to(device) for adj in adjs]
        feature, order0 = feature
        label, order1 = label
        feature = torch.cat([f.to(device) for f in feature])
        label = torch.cat([l.to(device) for l in label])
        origin_feature = torch.empty_like(feature)
        origin_label = torch.empty_like(label)
        origin_feature[order0] = feature
        origin_label[order1] = label

        optimizer.zero_grad()
        out = model(origin_feature, adjs)
        loss = F.nll_loss(out, origin_label)
        loss.backward()
        if sync.reduce:
            t0 = time.time()
            optimizer.sync_gradients()
            rd += time.time() - t0
            optimizer.pure_step()
        else:
            optimizer.step()
    print('cpu count: ' + str(cpu_count))
    print('gpu count: ' + str(gpu_count))
    print('train wait: ' + str(wait))
    print('allreduce: ' + str(rd))
    print('queue size: ' + str(sum(qs) / len(qs)))
    print(qs)
    sync.barriers[-1].wait()


class Trainer:
    def __init__(self, f, *args):
        self.f = f
        self.args = args

    def __call__(self, rank):
        rank = rank % 4
        self.f(rank, *self.args)


def main():
    cpu_num = 0
    gpu_num = 4
    train_num = 4
    device_num = 4
    dist = True
    home = os.getenv('HOME')
    data_dir = osp.join(home, '.pyg')
    root = osp.join(data_dir, 'data', 'products')
    dataset = PygNodePropPredDataset('ogbn-products', root)
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    train_idx = split_idx['train']

    queues = []
    for i in range(4):
        queues.append(mp.Queue(64))
    num = gpu_num + train_num + 1
    if cpu_num > 0:
        num += 1
    barrier1 = mp.Barrier(num)
    barrier2 = mp.Barrier(train_num + 1)
    sync = SyncConfig(0, device_num, queues, [barrier1, barrier2], 3,
                      train_num > 1, dist)
    x, y = data.x, data.y.squeeze()
    if cpu_num > 0:
        dev = ('cpu', cpu_num)
        sync.rank = i
        sync.dist = False
        cpu = mp.Process(target=SamplerProcess,
                         args=(sync, dev, data.edge_index, train_idx,
                               [15, 10, 5], 1024))
        cpu.start()
    sync.dist = dist
    for i in range(gpu_num):
        sync.rank = i
        dev = (i % sync.world_size, i // sync.world_size)
        gpu0 = mp.Process(target=SamplerProcess,
                          args=(sync, dev, data.edge_index, (x, y), train_idx,
                                [15, 10, 5], 1024))
        gpu0.start()
        time.sleep(1)
    if train_num == 1:
        train = mp.Process(target=TrainProcess,
                           args=(0, sync, 192, dataset.num_features, 256,
                                 dataset.num_classes, 3))
        train.start()
    else:
        train = Trainer(TrainProcess, sync, 192, dataset.num_features, 256,
                        dataset.num_classes, 3)
        trainers = launch_multiprocess(train, train_num)
    barrier1.wait()
    t0 = time.time()
    barrier2.wait()
    print(f'end to end {time.time() - t0}')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
