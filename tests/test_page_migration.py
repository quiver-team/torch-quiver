import cupy as cp
import numpy as np
import torch
import scipy.sparse as sp
import torch_quiver as qv
import time
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from scipy.sparse import csr_matrix
from torch.profiler import profile, record_function, ProfilerActivity

pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)

def test_neighbor_sampler_with_real_graph():
    print(f"{'*' * 10} TEST WITH REAL GRAPH {'*' * 10}")
    root = "/home/dalong/data/"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    data = dataset[0]
    x = data.x
    shape = x.shape
    array = x.numpy()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        def fn():
            x_numa = cp.random.random(shape, dtype=np.float32)
            x = torch.as_tensor(x_numa)
        fn()
    
test_neighbor_sampler_with_real_graph