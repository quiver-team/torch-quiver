import torch
import torch.multiprocessing as mp
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborSampler
from torch_geometric.datasets import Reddit
import time
import numpy as np
import os.path as osp
import random
from multiprocessing.reduction import ForkingPickler
import quiver
#from sampler import MixedGraphSageSampler, GraphSageSampler

print(
    "\n\nNOTE: We Use Sampled Edges Per Second(SEPS) = #SampledEdges/Time as metric to evaluate sampler performance\n\n"
)


def bench_on_ogbproduct():
    print("=" * 20 + "OGBn-Product Gpex" + "=" * 20)
    root = "/home/dalong/data/products"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    train_idx = dataset.get_idx_split()["train"]
    train_loader = torch.utils.data.DataLoader(train_idx,
                                               batch_size=256,
                                               pin_memory=True,
                                               shuffle=True)
    csr_topo = quiver.CSRTopo(dataset[0].edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5],
                                                 device=0,
                                                 mode="UVA")

    sample_time = 0
    sampled_edges = 0
    for seeds in train_loader:
        seeds = seeds.to(0)
        torch.cuda.synchronize()
        sample_start = time.time()
        _, _, adjs = quiver_sampler.sample(seeds)
        torch.cuda.synchronize()
        sample_time += time.time() - sample_start
        for adj in adjs:
            sampled_edges += adj.edge_index.shape[1]
    print(
        f"mean degree {np.mean(csr_topo.degree.numpy())}\tSample Speed {sampled_edges / sample_time / 1000000}M SEPS"
    )


def bench_on_ogbproduct_cpu():
    print("=" * 20 + "OGBn-Product CPU" + "=" * 20)
    root = "/data/products"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    train_idx = dataset.get_idx_split()["train"]
    train_loader = NeighborSampler(dataset[0].edge_index,
                                   node_idx=train_idx,
                                   sizes=[15, 10, 5],
                                   batch_size=1024,
                                   shuffle=True)

    sample_time = 0
    sampled_edges = 0
    sample_start = time.time()
    for batch_size, n_id, adjs in train_loader:
        sample_time += time.time() - sample_start
        for adj in adjs:
            sampled_edges += adj.edge_index.shape[1]
        sample_start = time.time()
    print(f"Sample Speed {sampled_edges / sample_time / 1000000}M SEPS")


def bench_on_reddit():
    print("=" * 20 + "Reddit" + "=" * 20)
    dataset = Reddit('/data/data/Reddit')
    train_mask = dataset[0].train_mask
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    train_loader = torch.utils.data.DataLoader(train_idx,
                                               batch_size=1024,
                                               pin_memory=True,
                                               shuffle=True)
    csr_topo = quiver.CSRTopo(dataset[0].edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [25, 10],
                                                 device=0,
                                                 mode="UVA")

    sample_time = 0
    sampled_edges = 0
    for seeds in train_loader:
        seeds = seeds.to(0)
        torch.cuda.synchronize()
        sample_start = time.time()
        _, _, adjs = quiver_sampler.sample(seeds)
        torch.cuda.synchronize()
        sample_time += time.time() - sample_start
        for adj in adjs:
            sampled_edges += adj.edge_index.shape[1]
    print(
        f"mean degree {np.mean(csr_topo.degree.numpy())}\tSample Speed {sampled_edges / sample_time / 1000000}M SEPS"
    )


def bench_on_reddit_cpu():
    print("=" * 20 + "Reddit CPU" + "=" * 20)
    dataset = Reddit('/data/data/Reddit')
    train_mask = dataset[0].train_mask
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    train_loader = NeighborSampler(dataset[0].edge_index,
                                   node_idx=train_idx,
                                   sizes=[25, 10],
                                   batch_size=1024,
                                   shuffle=True)

    sample_start = time.time()
    sample_time = 0
    sampled_edges = 0
    for batch_size, n_id, adjs in train_loader:
        sample_time += time.time() - sample_start
        for adj in adjs:
            sampled_edges += adj.edge_index.shape[1]
        sample_start = time.time()
    print(f"Sample Speed {sampled_edges / sample_time / 1000000}M SEPS")


def bench_child(rank, train_idx, indptr, indices, mode):
    torch.cuda.set_device(rank)
    train_loader = torch.utils.data.DataLoader(train_idx,
                                               batch_size=1024,
                                               pin_memory=True,
                                               shuffle=True)
    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5],
                                                 device=rank,
                                                 mode=mode)

    sampled_edges = 0
    sample_time = 0
    for seeds in train_loader:
        sample_start = time.time()
        _, _, adjs = quiver_sampler.sample(seeds)
        sample_time += time.time() - sample_start
        for adj in adjs:
            sampled_edges += adj.edge_index.shape[1]
    print(
        f"mean degree {np.mean(csr_topo.degree.to('cpu').numpy())}\tSample Speed {sampled_edges / sample_time / 1000000}M SEPS"
    )


def bench_on_ogbproduct_dist():
    print("=" * 20 + "OGBn-Product Gpex" + "=" * 20)
    root = "/data/products"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    train_idx = dataset.get_idx_split()["train"]
    csr_topo = quiver.CSRTopo(dataset[0].edge_index)

    mode = "UVA"
    world_size = 1
    procs = []
    for i in range(world_size):
        proc = mp.Process(target=bench_child,
                          args=(i, train_idx, csr_topo.indptr,
                                csr_topo.indices, mode))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()


def bench_on_paper100M_dist():
    print("=" * 20 + "Paper100M" + "=" * 20)
    root = "/data/papers/"

    data_dir = osp.join(root, 'ogbn_papers100M')
    indptr_root = osp.join(data_dir, 'csr', 'indptr.pt')
    indices_root = osp.join(data_dir, 'csr', 'indices.pt')
    index_root = osp.join(data_dir, 'index', 'train_idx.pt')

    train_idx = torch.load(index_root)
    indptr = torch.load(indptr_root)
    indices = torch.load(indices_root)

    mode = "UVA"
    world_size = 1
    procs = []
    for i in range(world_size):
        proc = mp.Process(target=bench_child,
                          args=(i, train_idx, indptr, indices, mode))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()

class MySampleJob(quiver.SampleJob):
    def __init__(self, seeds, batch_size):
        self.seeds = seeds
        self.batch_size = batch_size
    
    def __getitem__(self, index):
        start = self.batch_size * index
        return self.seeds[start: start + self.batch_size]
    
    def shuffle(self):
        random.shuffle(self.seeds)
    
    def __len__(self):
        return self.seeds.shape[0] // self.batch_size

def bench_on_ogbproduct_mixed():
    print("=" * 20 + "OGBn-Product Mixed" + "=" * 20)
    root = "/data/papers/"

    data_dir = osp.join(root, 'ogbn_papers100M')
    indptr_root = osp.join(data_dir, 'csr', 'indptr.pt')
    indices_root = osp.join(data_dir, 'csr', 'indices.pt')
    index_root = osp.join(data_dir, 'index', 'train_idx.pt')

    train_idx = torch.load(index_root)
    indptr = torch.load(indptr_root)
    indices = torch.load(indices_root)
    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    '''
    root = "/data/products"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    train_idx = dataset.get_idx_split()["train"]
    csr_topo = quiver.CSRTopo(dataset[0].edge_index)
    '''
    # batch_size = 256
    sample_job = MySampleJob(train_idx, 1024)
    print(f"Job task num = ", len(sample_job))
    
    # num_workers = 10
    # mode in ['GPU_ONLY', 'GPU_CPU_MIXED']
    # mode in ['UVA_ONLY', 'UVA_CPU_MIXED']

    quiver_sampler = quiver.MixedGraphSageSampler(sample_job, 3, csr_topo, [15, 10, 5], device=0, mode="UVA_CPU_MIXED")

    sample_time = 0
    sampled_edges = 0
    sample_start = time.time()
    for epoch in range(3):
        for _, res in enumerate(quiver_sampler):
            if epoch == 2:
                sample_time += time.time() - sample_start
                _, _, adjs = res
                for adj in adjs:
                    sampled_edges += adj.edge_index.shape[1]
            # Simulate Feature Collection + Train 
            time.sleep(0.030)
            sample_start = time.time()
        print(f"Epoch:{epoch} Finished")
    print(
        f"mean degree {np.mean(csr_topo.degree.numpy())}\tSample Speed {sampled_edges / sample_time / 1000000}M SEPS"
    )

if __name__ == "__main__":
    mp.set_start_method('spawn')
    #bench_on_ogbproduct()
    # bench_on_ogbproduct_cpu()
    # bench_on_reddit()
    # bench_on_reddit_cpu()
    # bench_on_paper100M()
    # bench_on_paper100M_dist()
    #bench_on_ogbproduct_dist()
    bench_on_ogbproduct_mixed()
