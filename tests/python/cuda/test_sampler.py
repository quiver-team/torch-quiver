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
from dataclasses import dataclass

__all__ = ["GraphSageSampler", "GraphStructure"]


class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size)

@dataclass(frozen=True)
class _FakeDevice(object):
    pass

class GraphSageSampler:
    r"""
    Quiver's GraphSageSampler behaves just like Pyg's `NeighborSampler` but with much higher performance.
    It can work in `UVA` mode or `GPU` mode. You can set `mode=GPU` if you have enough GPU memory to place graph's topology data which will offer the best sample performance.
    When your graph is too big for GPU memory, you can set `mode=UVA` to still use GPU to perform sample but place the data in host memory. `UVA` mode suffers 30%-40% performance loss compared to `GPU` mode
    but is much faster than CPU sampling(normally 16x~20x) and it consumes much less GPU memory compared to `GPU` mode.

    Args:
        csr_topo (quiver.CSRTopo): A quiver.CSRTopo for graph topology
        sizes ([int]): The number of neighbors to sample for each node in each
            layer. If set to `sizes[l] = -1`, all neighbors are included
            in layer `l`.
        device (int): Device which sample kernel will be launched
        mode (str): Sample mode, choices are [`UVA`, `GPU`, `CPU`, `GPU_CPU_MIXED`, `UVA_CPU_MIXED`], default is `UVA`.
    """
    def __init__(self,
                 csr_topo: quiver_utils.CSRTopo,
                 sizes: List[int],
                 device = 0,
                 mode="UVA"):

        assert mode in ["UVA",
                        "GPU",
                        "CPU"], f"sampler mode should be one of [UVA, GPU]"
        assert device is _FakeDevice or (device >= 0 and mode != "CPU") or (device < 0 and mode == "CPU"), f"Device setting and Mode setting not compatitive"
        
        self.sizes = sizes
        self.quiver = None
        self.csr_topo = csr_topo
        self.mode = mode
        if self.mode in ["GPU", "UVA"] and device is not _FakeDevice and  device >= 0:
            edge_id = torch.zeros(1, dtype=torch.long)
            self.quiver = qv.device_quiver_from_csr_array(self.csr_topo.indptr,
                                                       self.csr_topo.indices,
                                                       edge_id, device,
                                                       self.mode != "UVA")
        elif self.mode == "CPU" and device is not _FakeDevice:
            self.quiver = qv.cpu_quiver_from_csr_array(self.csr_topo.indptr, self.csr_topo.indices)
            device = "cpu"
        
        self.device = device
        self.ipc_handle_ = None

    def sample_layer(self, batch, size):
        self.lazy_init_quiver()
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)
        n_id = batch.to(self.device)
        size = size if size != -1 else self.csr_topo.node_count
        if self.mode in ["GPU", "UVA"]:
            n_id, count = self.quiver.sample_neighbor(0, n_id, size)
        else:
            n_id, count = self.quiver.sample_neighbor(n_id, size)
            
        return n_id, count

    def lazy_init_quiver(self):

        if self.quiver is not None:
            return

        self.device = "cpu" if self.mode == "CPU" else torch.cuda.current_device()
        
    
        if "CPU"  == self.mode:
            self.quiver = qv.cpu_quiver_from_csr_array(self.csr_topo.indptr, self.csr_topo.indices)
        else:
            edge_id = torch.zeros(1, dtype=torch.long)
            self.quiver = qv.device_quiver_from_csr_array(self.csr_topo.indptr,
                                                       self.csr_topo.indices,
                                                       edge_id, self.device,
                                                       self.mode != "UVA")

    def reindex(self, inputs, outputs, counts):
        return self.quiver.reindex_single(inputs, outputs, counts)

    def sample(self, input_nodes):
        """Sample k-hop neighbors from input_nodes

        Args:
            input_nodes (torch.LongTensor): seed nodes ids to sample from

        Returns:
            Tuple: Return results are the same with Pyg's sampler
        """
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
        """Create ipc handle for multiprocessing

        Returns:
            tuple: ipc handle tuple
        """
        return self.csr_topo, self.sizes, self.mode

    @classmethod
    def lazy_from_ipc_handle(cls, ipc_handle):
        """Create from ipc handle

        Args:
            ipc_handle (tuple): ipc handle got from calling `share_ipc`

        Returns:
            quiver.pyg.GraphSageSampler: Sampler created from ipc handle
        """
        csr_topo, sizes, mode = ipc_handle
        return cls(csr_topo, sizes, _FakeDevice, mode)


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
    return sampler


def reduce_pyg_sampler(sampler):
    print("reduce sampler")
    ipc_handle = sampler.share_ipc()
    return (rebuild_pyg_sampler, (
        type(sampler),
        ipc_handle,
    ))


def init_reductions():
    ForkingPickler.register(GraphSageSampler, reduce_pyg_sampler)

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

if __name__ == "__main__":
    mp.set_start_method("spawn")
    init_reductions()

    #test_GraphSageSampler()
    #test_ipc()
    test_cpu_mode()
