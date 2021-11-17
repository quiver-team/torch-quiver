import torch
import numpy as np
import scipy.sparse as sp
import torch_quiver as qv

import time
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from scipy.sparse import csr_matrix
import os
import os.path as osp
import quiver
import torch.multiprocessing as mp
from multiprocessing.reduction import ForkingPickler
import torch
from torch import Tensor
from torch_sparse import SparseTensor
import torch_quiver as qv
from typing import List, Optional, Tuple, NamedTuple, Union, Callable

import quiver.utils as quiver_utils

import torch
from torch import Tensor
from torch_sparse import SparseTensor
import torch_quiver as qv
from typing import List, Optional, Tuple, NamedTuple, Union, Callable
from dataclasses import dataclass
from sampler import MixedGraphSageSampler, GraphSageSampler, SampleJob
import random

def test_GraphSageSampler():
    """
    class GraphSageSampler:

    def __init__(self, edge_index: Union[Tensor, SparseTensor], sizes: List[int], device, num_nodes: Optional[int] = None, mode="UVA", device_replicate=True):
    """
    print(f"{'*' * 10} TEST WITH REAL GRAPH {'*' * 10}")
    #home = os.getenv('HOME')
    #ata_dir = osp.join(home, '.pyg')
    #root = osp.join(data_dir, 'data', 'products')
    root = "/data/data/products/"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    torch.cuda.set_device(0)
    data = dataset[0]

    seeds_size = 128 * 15 * 10
    neighbor_size = 5

    seeds = np.arange(2000000)
    np.random.shuffle(seeds)
    seeds = seeds[:seeds_size]
    seeds = torch.from_numpy(seeds).type(torch.long)
    cuda_seeds = seeds.to(0)

    csr_topo = quiver.CSRTopo(data.edge_index)

    sage_sampler = GraphSageSampler(csr_topo, sizes=[5], device=0, mode="GPU")
    res = sage_sampler.sample(cuda_seeds)
    print(res)


def child_process(rank, sage_sampler):

    torch.cuda.set_device(rank)
    seeds_size = 1024
    neighbor_size = 5
    node_count = sage_sampler.csr_topo.indptr.shape[0] - 1

    seeds = np.arange(node_count)
    np.random.shuffle(seeds)
    seeds = seeds[:seeds_size]
    seeds = torch.from_numpy(seeds).type(torch.long)
    cuda_seeds = seeds.to(rank)

    res = sage_sampler.sample(cuda_seeds)
    sample_times = []
    for _ in range(200):

        start = time.time()
        res = sage_sampler.sample(cuda_seeds)
        sample_times.append(time.time() - start)

    print(f"consumed {time.time() - start}")


def test_ipc():
    root = "/home/dalong/data/products/"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    torch.cuda.set_device(0)
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)
    sage_sampler = GraphSageSampler(csr_topo,
                                    sizes=[15, 10, 5],
                                    device=0,
                                    mode="GPU")

    mp.spawn(child_process, args=(sage_sampler, ), nprocs=1, join=True)


def rebuild_pyg_sampler(cls, ipc_handle):
    print("rebuild sampler")
    sampler = cls.lazy_from_ipc_handle(ipc_handle)
    print("rebuild successfully")
    return sampler


def reduce_pyg_sampler(sampler):
    print("reduce sampler")
    ipc_handle = sampler.share_ipc()
    return (rebuild_pyg_sampler, (
        type(sampler),
        ipc_handle,
    ))


def init_reductions():
    print("init reductions")
    ForkingPickler.register(GraphSageSampler, reduce_pyg_sampler)
    ForkingPickler.register(MixedGraphSageSampler, reduce_pyg_sampler)

def test_cpu_mode():
    print(f"{'*' * 10} TEST WITH REAL GRAPH {'*' * 10}")

    root = "/home/dalong/data/products/"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    data = dataset[0]

    seeds_size = 128 * 15 * 10
    neighbor_size = 5

    seeds = np.arange(2000000)
    np.random.shuffle(seeds)
    seeds = seeds[:seeds_size]
    seeds = torch.from_numpy(seeds).type(torch.long)

    csr_topo = quiver.CSRTopo(data.edge_index)

    sage_sampler = GraphSageSampler(csr_topo, sizes=[neighbor_size], device=-1, mode="CPU")
    print(csr_topo.indices[csr_topo.indptr[seeds[0]]: csr_topo.indptr[seeds[0] + 1]])
    res = sage_sampler.sample(seeds)
    print(res)

class MySampleJob(SampleJob):
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

def test_mixed_mode():
    root = "/home/dalong/data/products/"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    train_idx = dataset.get_idx_split()['train']
    sample_job = MySampleJob(train_idx, 64)

    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)
    sage_sampler = MixedGraphSageSampler(sample_job, 5, csr_topo, sizes=[15, 10, 5], device=0, mode="UVA_CPU_MIXED")
    for epoch in range(10):
        for res in sage_sampler:
            pass
        print("epoch finished")

def mixed_child_process(rank, sage_sampler):
    for epoch in range(2):
        for res in sage_sampler:
            pass
        print("epoch finished")

def test_mixed_mode_ipc():
    root = "/home/dalong/data/products/"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    train_idx = dataset.get_idx_split()['train']
    sample_job = MySampleJob(train_idx, 256)

    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)
    sage_sampler = quiver.pyg.MixedGraphSageSampler(sample_job, 5, csr_topo, sizes=[15, 10, 5], device=0, mode="UVA_CPU_MIXED")

    mp.spawn(mixed_child_process, args=(sage_sampler, ), nprocs=1, join=True)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    #init_reductions()

    #test_GraphSageSampler()
    #test_ipc()
    #test_cpu_mode()
    #test_mixed_mode()
    test_mixed_mode_ipc()
