import torch
import numpy as np 
import scipy.sparse as sp
import torch_quiver as qv

import time
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from scipy.sparse import csr_matrix
import os
import os.path as osp

from quiver.sage_sampler import GraphSageSampler


def test_GraphSageSampler():
    """
    class GraphSageSampler:

    def __init__(self, edge_index: Union[Tensor, SparseTensor], sizes: List[int], device, num_nodes: Optional[int] = None, mode="UVA", device_replicate=True):
    """
    print(f"{'*' * 10} TEST WITH REAL GRAPH {'*' * 10}")
    home = os.getenv('HOME')
    data_dir = osp.join(home, '.pyg')
    root = osp.join(data_dir, 'data', 'products')
    dataset = PygNodePropPredDataset('ogbn-products', root)
    data = dataset[0]
    edge_index = data.edge_index
    seeds_size = 128 * 15 * 10
    neighbor_size = 5
    
    graph_size = csr_mat.indptr.shape[0] - 1
    seeds = np.arange(graph_size)
    np.random.shuffle(seeds)
    seeds =seeds[:seeds_size]
    seeds = torch.from_numpy(seeds).type(torch.long)
    cuda_seeds = seeds.to(0)

    sage_sampler = GraphSageSampler(data.edge_index, sizes=[5], device=0, num_nodes=graph_size)
    res = sage_sampler(cuda_seeds)
    

test_GraphSageSampler()
