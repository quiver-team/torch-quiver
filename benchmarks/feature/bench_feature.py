import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Reddit
import time
import numpy as np
import os.path as osp

import quiver

def bench_on_ogbproduct():
    print("=" * 20  + "OGBn-Product" + "=" * 20)
    root = "/home/dalong/products"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    train_idx = dataset.get_idx_split()["train"]
    train_loader = torch.utils.data.DataLoader(train_idx, batch_size=1024, pin_memory=True, shuffle=True)
    csr_topo = quiver.CSRTopo(dataset[0].edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], device=0, mode="UVA")
    quiver_feature = quiver.Feature(rank=0, device_list=[0, 1], device_cache_size="200M", cache_policy="p2p_clique_replicate", csr_topo=csr_topo)
    feature = torch.zeros(dataset[0].x.shape)
    feature[:] = dataset[0].x
    quiver_feature.from_cpu_tensor(feature)
    accessed_feature_size = 0
    feature_time = 0
    for seeds in train_loader:
        nid, _, _ = quiver_sampler.sample(seeds)
        torch.cuda.synchronize()
        feature_start = time.time()
        res = quiver_feature[nid]
        torch.cuda.synchronize()
        feature_time += time.time() - feature_start
        accessed_feature_size += res.numel() * 4
    torch.cuda.synchronize()
    print(f"Feature Collection Throughput {accessed_feature_size / feature_time / 1024 / 1024 / 1024} GB/s")



def bench_on_reddit():
    dataset = Reddit('/home/dalong/data/Reddit')
    train_mask = dataset[0].train_mask
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    train_loader = torch.utils.data.DataLoader(train_idx, batch_size=1024, pin_memory=True, shuffle=True)
    csr_topo = quiver.CSRTopo(dataset[0].edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [25, 10], device=0, mode="UVA")
    quiver_feature = quiver.Feature(rank=0, device_list=[0, 1], device_cache_size="110M", cache_policy="p2p_clique_replicate", csr_topo=csr_topo)
    quiver_feature.from_cpu_tensor(dataset[0].x)
    accessed_feature_size = 0
    feature_time = 0
    for seeds in train_loader:
        nid, _, _ = quiver_sampler.sample(seeds)
        torch.cuda.synchronize()
        feature_start = time.time()
        res = quiver_feature[nid]
        torch.cuda.synchronize()
        feature_time += time.time() - feature_start
        accessed_feature_size += res.numel() * 4
    torch.cuda.synchronize()
    print(f"Feature Collection Throughput {accessed_feature_size / feature_time / 1024 / 1024 / 1024} GB/s")

def bench_on_paper100M():
    root = "/home/dalong/data/papers/"
    data_dir = osp.join(root, 'ogbn_papers100M')
    

if __name__ == "__main__":
    quiver.init_p2p([0, 1])
    #bench_on_reddit()
    bench_on_ogbproduct()
    #bench_on_paper100M()
