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
#from quiver.pyg import GraphSageSampler
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



__all__ = ["GraphSageSampler", "GraphStructure"]


class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size)
                   
class GraphSageSampler:
    r"""
    The graphsage sampler from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, which allows
    for mini-batch training of GNNs on large-scale graphs where full-batch
    training is not feasible.

    Args:
        csr_topo (quiver_utils.CSRTopo): A quiver_utils.CSRTopo
        sizes ([int]): The number of neighbors to sample for each node in each
            layer. If set to :obj:`sizes[l] = -1`, all neighbors are included
            in layer :obj:`l`.
        device (int): Device which sample kernel will be launched
        num_nodes (int, optional): The number of nodes in the graph.
            (default: :obj:`None`)
        mode (str): Sample mode, choices are [UVA, GPU].
            (default: :obj: `UVA`)
    """

    def __init__(self, csr_topo: quiver_utils.CSRTopo, sizes: List[int], device, mode="UVA"):

        
        self.sizes = sizes
        
        self.quiver = None
        self.csr_topo = csr_topo

        self.mode = mode
        if device >= 0:
            edge_id = torch.zeros(1, dtype=torch.long)
            self.quiver = qv.new_quiver_from_csr_array(self.csr_topo.indptr, self.csr_topo.indices, edge_id, device, self.mode != "UVA")
    
        self.device = device

        self.ipc_handle_ = None

    
    def sample_layer(self, batch, size):
        self.lazy_init_quiver()
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)
        n_id = batch.to(torch.device(self.device))
        n_id, count = self.quiver.sample_neighbor(0, n_id, size)
        return n_id, count

    def lazy_init_quiver(self):
        if self.quiver is not None:
            return 
        self.device = torch.cuda.current_device()
        edge_id = torch.zeros(1, dtype=torch.long)
        self.quiver = qv.new_quiver_from_csr_array(self.csr_topo.indptr, self.csr_topo.indices, edge_id, self.device, self.mode != "UVA")

    def reindex(self, inputs, outputs, counts):
        return qv.reindex_single(inputs, outputs, counts)

    def sample(self, input_nodes):
        self.lazy_init_quiver()
        nodes = input_nodes.to(self.device)
        adjs = []

        batch_size = len(nodes)
        for size in self.sizes:
            out, cnt = self.sample_layer(nodes, size)
            frontier, row_idx, col_idx = self.reindex(nodes, out, cnt)
            row_idx, col_idx = col_idx, row_idx
            edge_index = torch.stack([row_idx, col_idx], dim=0)

            adj_size = torch.LongTensor([
                frontier.size(0),
                nodes.size(0),
            ])
            e_id = torch.tensor([])
            adjs.append(Adj(edge_index, e_id, adj_size))
            nodes = frontier

        return nodes, batch_size, adjs[::-1]

    def share_ipc(self):
        return self.csr_topo, self.sizes, self.mode
    
    @classmethod
    def lazy_from_ipc_handle(cls, ipc_handle):
        csr_topo, sizes, mode = ipc_handle
        return cls(csr_topo, sizes, -1, mode)


def test_GraphSageSampler():
    """
    class GraphSageSampler:

    def __init__(self, edge_index: Union[Tensor, SparseTensor], sizes: List[int], device, num_nodes: Optional[int] = None, mode="UVA", device_replicate=True):
    """
    print(f"{'*' * 10} TEST WITH REAL GRAPH {'*' * 10}")
    #home = os.getenv('HOME')
    #ata_dir = osp.join(home, '.pyg')
    #root = osp.join(data_dir, 'data', 'products')
    root = "/home/dalong/data/products/"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    torch.cuda.set_device(0)
    data = dataset[0]

    seeds_size = 128 * 15 * 10
    neighbor_size = 5
    
    seeds = np.arange(2000000)
    np.random.shuffle(seeds)
    seeds =seeds[:seeds_size]
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
    seeds =seeds[:seeds_size]
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
    root = "/home/dalong/products/"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    torch.cuda.set_device(0)
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)
    sage_sampler = quiver.pyg.GraphSageSampler(csr_topo, sizes=[15, 10, 5], device=0, mode="GPU")

    mp.spawn(child_process, args=(sage_sampler), nprocs=1, join=True)
    
def rebuild_pyg_sampler(cls, ipc_handle):
    print("rebuild sampler")
    sampler = cls.lazy_from_ipc_handle(ipc_handle)
    return sampler
    

def reduce_pyg_sampler(sampler):
    print("reduce sampler")
    ipc_handle = sampler.share_ipc()
    return (rebuild_pyg_sampler, (type(sampler), ipc_handle, ))
  

def init_reductions():
    ForkingPickler.register(GraphSageSampler, reduce_pyg_sampler)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    init_reductions()

    #test_GraphSageSampler()
    test_ipc()
