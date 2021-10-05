from ogb.nodeproppred import PygNodePropPredDataset
from kungfu.cmd import launch_multiprocess
import argparse
import multiprocessing as mp
import os
import os.path as osp
import time

import queue

import psutil

import torch
import torch.nn.functional as F
from torch.distributed import rpc

from quiver.cuda_sampler import CudaNeighborSampler
from quiver.schedule.policy import Policy, PolicyFilter
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
from tqdm import tqdm


class ProductsDataset:
    def __init__(self, train_idx, edge_index, x, y, f, c):
        self.train_idx = train_idx
        self.edge_index = edge_index
        self.x = x
        self.y = y
        self.num_features = f
        self.num_classes = c


class SyncConfig:
    def __init__(self, group, rank, world_size, queues, barriers, timeout,
                 reduce, dist):
        self.group = group
        self.rank = rank
        self.world_size = world_size
        self.queues = queues
        self.barriers = barriers
        self.timeout = timeout
        self.reduce = reduce
        self.dist = dist


num_device = torch.cuda.device_count()
train_thread = psutil.cpu_count() // num_device
sample_thread = psutil.cpu_count() // num_device
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
    group = sync.group
    sync.rank = sync.rank % sync.world_size
    torch.set_num_threads(sample_thread)

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
            local_rank = sync.rank
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
        # torch.set_num_threads(1)
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
                     backend=rpc.BackendType.TENSORPIPE,
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
            try:
                t0 = time.time()
                if count > 1:
                    sync.queues[0].put((device, sample), timeout=sync.timeout)
                wait += time.time() - t0
            except queue.Full:
                cont = False
                break
    # print(dev)
    # print('sample wait: ' + str(wait))
    time.sleep(30)


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


def TrainProcess(rank, sync, data, num_batch, num_features, num_hidden,
                 num_classes, num_layers):
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
    x, y = data
    torch.set_num_threads(train_thread)
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
        batch_size, n_id, adjs = sample
        adjs = [adj.to(device) for adj in adjs]
        feature = x[n_id].to(device)
        label = y[n_id[:batch_size]].to(device)

        optimizer.zero_grad()
        out = model(feature, adjs)
        loss = F.nll_loss(out, label)
        loss.backward()
        if sync.reduce:
            t0 = time.time()
            optimizer.sync_gradients()
            rd += time.time() - t0
            optimizer.pure_step()
        else:
            optimizer.step()
    # print('cpu count: ' + str(cpu_count))
    # print('gpu count: ' + str(gpu_count))
    # print('train wait: ' + str(wait))
    # print('allreduce: ' + str(rd))
    # print('queue size: ' + str(sum(qs) / len(qs)))
    # print(qs)
    sync.barriers[-1].wait()


class Trainer:
    def __init__(self, f, ws, *args):
        self.f = f
        self.ws = ws
        self.args = args

    def __call__(self, rank):
        rank = self.ws - rank - 1
        self.f(rank, *self.args)


def main(policy, num_batch=64, use_products=True):
    train_num = policy.num_train
    device_num = policy.num_dev
    cpu_num = policy.num_cpu
    batch_size = policy.batch_size
    dist = True
    if use_products:
        root = './products.pt'
        dataset = torch.load(root)
        train_idx = dataset.train_idx
        edge_index = dataset.edge_index
        x, y = dataset.x.share_memory_(), dataset.y.squeeze().share_memory_()
    else:
        home = os.getenv('HOME')
        data_dir = osp.join(home, '.pyg')
        root = osp.join(data_dir, 'data', 'products')
        dataset = PygNodePropPredDataset('ogbn-products', root)
        split_idx = dataset.get_idx_split()
        data = dataset[0]
        train_idx = split_idx['train']
        edge_index = data.edge_index
        x, y = data.x.share_memory_(), data.y.squeeze().share_memory_()

    queues = [mp.Queue(16), mp.Queue(8)]
    num = -train_num * policy.count_sub() + device_num * \
        policy.count() + train_num + 1
    if cpu_num > 0:
        num += 1
    barrier1 = mp.Barrier(num)
    barrier2 = mp.Barrier(train_num + 1)
    group = 0
    samplers = []
    for i in range(policy.count_sub()):
        for j in range(device_num - train_num):
            dev = (j, i)
            sync = SyncConfig(group, j, device_num - train_num, queues,
                              [barrier1, barrier2], 3, train_num > 1, dist)
            gpu0 = mp.Process(target=SamplerProcess,
                              args=(sync, dev, edge_index, (None, None),
                                    train_idx, [15, 10, 5], batch_size))
            gpu0.start()
            samplers.append(gpu0)
            time.sleep(1)
        group += 1
    for i in range(policy.count() - policy.count_sub()):
        for j in range(device_num):
            dev = (j, i)
            sync = SyncConfig(group, j, device_num, queues,
                              [barrier1, barrier2], 3, train_num > 1, dist)
            gpu0 = mp.Process(target=SamplerProcess,
                              args=(sync, dev, edge_index, (None, None),
                                    train_idx, [15, 10, 5], batch_size))
            gpu0.start()
            samplers.append(gpu0)
            time.sleep(1)
        group += 1

    if cpu_num > 0:
        dev = ('cpu', cpu_num)
        sync = SyncConfig(0, 0, device_num, queues, [barrier1, barrier2], 3,
                          train_num > 1, False)
        cpu = mp.Process(target=SamplerProcess,
                         args=(sync, dev, edge_index, (None, None), train_idx,
                               [15, 10, 5], batch_size))
        cpu.start()
        samplers.append(cpu)

    sync = SyncConfig(group, 0, device_num, queues, [barrier1, barrier2], 3,
                      train_num > 1, dist)
    if train_num == 1:
        train = mp.Process(target=TrainProcess,
                           args=(device_num - 1, sync, (x, y), num_batch,
                                 dataset.num_features, 256,
                                 dataset.num_classes, 3))
        train.start()
        trainers = [train]
    else:
        train = Trainer(TrainProcess, device_num, sync, (x, y), num_batch,
                        dataset.num_features, 256, dataset.num_classes, 3)
        trainers = launch_multiprocess(train, train_num)
    barrier1.wait()
    t0 = time.time()
    time.sleep(2)
    per = psutil.cpu_percent(1)
    barrier2.wait()
    qs = queues[0].qsize()
    dur = time.time() - t0
    # print(f'end to end {res}')
    for sampler in samplers:
        sampler.kill()
    for trainer in trainers:
        trainer.kill()
    return dur, per, qs


if __name__ == '__main__':
    mp.set_start_method('spawn')
    batch_size = 512
    cpu_num = psutil.cpu_count() // num_device
    print(f'cpu worker {cpu_num}')
    normal = Policy(batch_size, num_device, num_device, cpu_num)
    pyg = Policy(batch_size, num_device, num_device, cpu_num)
    pyg.remove_group()
    pf = PolicyFilter(batch_size, num_device, 0)
    for policy in pf:
        print(f'trainer {policy.num_train} group {policy.num_group}')
        stats = main(policy)
        pf.set_stats(stats)
        print(f'time {stats}')
    policy, stats = pf.best_group()
    print(f'best trainer {policy.num_train} group {policy.num_group}')
    stats = main(policy)
    print('finish base')
    dur, per, qs = stats
    res_all = 999.99, 0, 0
    res_sub = 999.99, 0, 0
    res_cpu = 999.99, 0, 0
    if qs <= policy.count() * num_device - policy.count_sub(
    ) * policy.num_train:
        if policy.count_sub() > 0:
            policy.add_sub_group()
            res_sub = main(policy)
            print('finish sub')
            policy.remove_sub_group()
            stats = min((stats, res_sub))
        policy.num_cpu = int((100 - per) / 100 * psutil.cpu_count())
        policy.num_cpu = min(policy.num_cpu, cpu_num)
        print(f'cpu num {policy.num_cpu}')
        res_cpu = main(policy)
        print('finish cpu')
        stats = min((stats, res_cpu))
        if policy.count_sub() > 0:
            policy.add_sub_group()
            res_all = main(policy)
            stats = min((stats, res_all))
        if stats == res_sub:
            policy.num_cpu = 0
            print('sub win')
        if stats == res_cpu:
            policy.remove_sub_group()
            print('cpu win')
        if stats == res_all:
            print('all win')
    print(str(policy))
    time.sleep(5)
    cpu, _, _, = main(pyg, 192)
    print(f'pyg finish {cpu}')
    time.sleep(5)
    tuned, _, _ = main(policy, 192)
    print(f'tuned finish {tuned}')
    time.sleep(5)
    norm, _, _, = main(normal, 192)
    print(f'norm finish {norm}')
    print(f'cpu {cpu} vs norm {norm} vs tuned {tuned}')
