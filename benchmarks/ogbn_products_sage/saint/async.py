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

from quiver.saint_sampler import  CudaRWSampler
from quiver.models.saint_model import Net
from torch_geometric.data import GraphSAINTRandomWalkSampler
import torch_quiver as qv

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

def sample_cuda(nodes, walk_length):
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:' + str(local_rank))
    cu_nodes = loader.global2local(nodes).to(device)
    node_idx = loader.adj.random_walk(cu_nodes, walk_length)[:,1]
    return  node_idx.to(torch.device('cpu'))

def sample_cpu(nodes, walk_length):
    node_idx = loader.adj.random_walk(nodes, walk_length)[:,1]
local_rank = None

def subgraph_cuda(nodes, i):
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:' + str(local_rank))
    adj_row, adj_col, _ = loader.adj.coo()
    adj_rowptr = loader.adj.storage.rowptr()
    row, col, edge_index = qv.saint_subgraph(nodes.to(device), adj_rowptr, adj_row, adj_col)
    cpu = torch.device('cpu')
    return row.to(cpu), col.to(cpu), edge_index.to(cpu)

def subgraph_cpu(nodes, i):
    adj, edge_index = loader.adj.saint_subgraph(nodes)
    return adj, edge_index

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
    torch.set_num_threads(5)

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
            import quiver.dist_saint_cuda_sampler as dist
            comm = dist.Comm(sync.rank, sync.world_size)
            local_rank = sync.rank % 4
            torch.cuda.set_device(device)
            loader = dist.distributeCudaRWSampler(
                comm, (data, local2global, global2local, node2rank),
                node_f, device,
                batch_size = batch_size, walk_length=2,
                num_steps=10,
                sample_coverage=0,
                save_dir="./cuda/")
            dist.sample_nodes = sample_cuda
            dist.subgraph_nodes = subgraph_cuda
        else:
            loader = CudaRWSampler(data,
                                   batch_size = batch_size,
                                   walk_length=2,
                                   num_steps=10,
                                   sample_coverage=0,
                                   save_dir="./cuda/")
    else:
        if sync.dist:
            import quiver.dist_saint_cpu_sampler as dist
            comm = dist.Comm(sync.rank, sync.world_size)
            local_rank = sync.rank % 4
            loader = dist.distributeRWSampler(
                comm, (data, local2global, global2local, node2rank),
                node_f,
                batch_size=batch_size, walk_length=2,
                num_steps=10,
                sample_coverage=0,
                save_dir="./cpu/")
            dist.sample_nodes = sample_cuda
            dist.subgraph_nodes = subgraph_cpu
        else:
            loader = GraphSAINTRandomWalkSampler(data,
                                                 batch_size = batch_size,
                                                 walk_length = 2,
                                                 num_steps = 10,
                                                 sample_coverage=0,
                                                 save_dir="./cpu/",
                                                 num_workers=num)
    if sync.dist:
        os.environ['MASTER_ADDR'] = 'localhost'
        if device == 'cpu':
            group += 32
        port = str(29500 + 100 * group)
        os.environ['MASTER_PORT'] = port
        print("in init rpc the sync rank", sync.rank)
        print("world size is ", sync.world_size)
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
    time.sleep(1)
    print(dev)
    print('sample wait: ' + str(wait))




def TrainProcess(rank, sync, data, num_batch, num_features, num_hidden,
                 num_classes, num_layers):
    if sync.reduce:
        import kungfu.torch as kf
        from kungfu.python import current_cluster_size, current_rank
    print("intrain", rank)
    device = torch.device('cuda:' +
                          str(rank) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(rank)
    model = Net(hidden_channels=num_hidden,
                num_node_features=num_features,
                num_classes=num_classes)
    model = model.to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    sync.rank = rank
    torch.set_num_threads(5)
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
        one_sample = sample
        feature = one_sample.x.to(device)
        label = one_sample.y.to(device)
        model.set_aggr('mean')
        optimizer.zero_grad()
        out = model(feature, one_sample.edge_index.to(device))
        loss = F.nll_loss(out[one_sample.train_mask], label[one_sample.train_mask].squeeze_())
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


def main(num_batch, cpu_num, gpu_num, train_num, device_num, batch_size):
    dist = True
    home = os.getenv('HOME')
    data_dir = osp.join(home, '.pyg')
    root = osp.join(data_dir, 'data', 'products')
    dataset = PygNodePropPredDataset('ogbn-products', root)
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    train_idx = split_idx['train']
    split_idx = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[split_idx['train']] = True
    data.train_mask = train_mask

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
    x, y = data.x.share_memory_(), data.y.squeeze().share_memory_()
    samplers = []
    if cpu_num > 0:
        dev = ('cpu', cpu_num)
        sync.rank = i
        sync.dist = True
        cpu = mp.Process(target=SamplerProcess,
                         args=(sync, dev, data.edge_index,data,
                               train_idx, [], batch_size))
        cpu.start()
        samplers.append(cpu)
    sync.dist = dist
    for i in range(gpu_num):
        sync.rank = i
        dev = (i % sync.world_size, i // sync.world_size)
        gpu0 = mp.Process(target=SamplerProcess,
                          args=(sync, dev, data.edge_index, data,
                                train_idx, [], batch_size))
        gpu0.start()
        samplers.append(gpu0)
        time.sleep(1)
    if train_num == 1:
        train = mp.Process(target=TrainProcess,
                           args=(0, sync, (x,
                                           y), num_batch, dataset.num_node_features,
                                 256, dataset.num_classes, 3))
        train.start()
        trainers = [train]
    else:
        train = Trainer(TrainProcess, sync, (x, y), 192, dataset.num_node_features,
                        256, dataset.num_classes, 3)
        trainers = launch_multiprocess(train, train_num)
    barrier1.wait()
    t0 = time.time()
    barrier2.wait()
    print(f'end to end {time.time() - t0}')
    for sampler in samplers:
        sampler.kill()
    for trainer in trainers:
        trainer.kill()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main(1800, 2, 0, 4, 4, 24000)
