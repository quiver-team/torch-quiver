import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborSampler
from torch_geometric.datasets import Reddit
import time
import numpy as np
import os.path as osp

import quiver

print("\n\nNOTE: We Use Sampled Edges Per Second(SEPS) = #SampledEdges/Time as metric to evaluate sampler performance\n\n")

def bench_on_ogbproduct():
    print("=" * 20  + "OGBn-Product Gpex" + "=" * 20)
    root = "/home/dalong/data/products"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    train_idx = dataset.get_idx_split()["train"]
    train_loader = torch.utils.data.DataLoader(train_idx, batch_size=1024, pin_memory=True, shuffle=True)
    csr_topo = quiver.CSRTopo(dataset[0].edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], device=0, mode="UVA")

    sample_start = time.time()
    sampled_edges = 0
    for seeds in train_loader:
        _, _, adjs = quiver_sampler.sample(seeds)
        for adj in adjs:
            sampled_edges += adj.edge_index.shape[1]
    print(f"mean degree {np.mean(csr_topo.degree.numpy())}\tSample Speed {sampled_edges / (time.time() - sample_start) / 1000000}M SEPS")

def bench_on_ogbproduct_cpu():
    print("=" * 20  + "OGBn-Product CPU" + "=" * 20)
    root = "/home/dalong/data/products"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    train_idx = dataset.get_idx_split()["train"]
    train_loader = NeighborSampler(dataset[0].edge_index, node_idx=train_idx, sizes=[15, 10, 5], batch_size=1024, shuffle=True)

    sample_start = time.time()
    sampled_edges = 0
    for batch_size, n_id, adjs in train_loader:
        for adj in adjs:
            sampled_edges += adj.edge_index.shape[1]
    print(f"Sample Speed {sampled_edges / (time.time() - sample_start) / 1000000}M SEPS")


def bench_on_reddit():
    print("=" * 20  + "Reddit" + "=" * 20)
    dataset = Reddit('/home/dalong/data/Reddit')
    train_mask = dataset[0].train_mask
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    train_loader = torch.utils.data.DataLoader(train_idx, batch_size=1024, pin_memory=True, shuffle=True)
    csr_topo = quiver.CSRTopo(dataset[0].edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [25, 10], device=0, mode="UVA")

    sample_start = time.time()
    sampled_edges = 0
    for seeds in train_loader:
        _, _, adjs = quiver_sampler.sample(seeds)
        for adj in adjs:
            sampled_edges += adj.edge_index.shape[1]
    print(f"mean degree {np.mean(csr_topo.degree.numpy())}\tSample Speed {sampled_edges / (time.time() - sample_start) / 1000000}M SEPS")

def bench_on_reddit_cpu():
    print("=" * 20  + "Reddit CPU" + "=" * 20)
    dataset = Reddit('/home/dalong/data/Reddit')
    train_mask = dataset[0].train_mask
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    train_loader = NeighborSampler(dataset[0].edge_index, node_idx=train_idx, sizes=[15, 10, 5], batch_size=1024, shuffle=True)

    sample_start = time.time()
    sampled_edges = 0
    for batch_size, n_id, adjs in train_loader:
        for adj in adjs:
            sampled_edges += adj.edge_index.shape[1]
    print(f"Sample Speed {sampled_edges / (time.time() - sample_start) / 1000000}M SEPS")


def bench_on_paper100M():
    print("=" * 20  + "Paper100M" + "=" * 20)
    root = "/home/dalong/data/papers/"
    data_dir = osp.join(root, 'ogbn_papers100M')
    indptr_root = osp.join(data_dir, 'csr', 'indptr.pt')
    indices_root = osp.join(data_dir, 'csr', 'indices.pt') 
    index_root = osp.join(data_dir, 'index', 'train_idx.pt')
    

    train_idx = torch.load(index_root)
    indptr = torch.load(indptr_root)
    indices = torch.load(indices_root)

    train_loader = torch.utils.data.DataLoader(train_idx, batch_size=1024, pin_memory=True, shuffle=True)
    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], device=0, mode="UVA")

    sample_start = time.time()
    sampled_edges = 0
    for seeds in train_loader:
        _, _, adjs = quiver_sampler.sample(seeds)
        for adj in adjs:
            sampled_edges += adj.edge_index.shape[1]
    print(f"mean degree {np.mean(csr_topo.degree.numpy())}\tSample Speed {sampled_edges / (time.time() - sample_start) / 1000000}M SEPS")

if __name__ == "__main__":
    #bench_on_ogbproduct()
    #bench_on_ogbproduct_cpu()
    bench_on_reddit()
    bench_on_reddit_cpu()
