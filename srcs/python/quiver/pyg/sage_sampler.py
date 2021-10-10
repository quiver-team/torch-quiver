import torch
from torch import Tensor
from torch_sparse import SparseTensor
import torch_quiver as qv
from typing import List, Optional, Tuple, NamedTuple, Union, Callable

from .. import utils as quiver_utils


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
        device_replicate: (bool): If replicate edge index for each device
            (default: :obj: `True`)
    """

    def __init__(self, csr_topo: quiver_utils.CSRTopo, sizes: List[int], device, mode="UVA", device_replicate=False):

        
        self.sizes = sizes
        
        self.quiver = None
        self.csr_topo = csr_topo

        self.mode = mode
        if device >= 0 and self.mode == "UVA":
            edge_id = torch.zeros(1, dtype=torch.long)
            self.quiver = qv.new_quiver_from_csr_array(self.csr_topo.indptr, self.csr_topo.indices, edge_id, device, device_replicate)
        else:
            pass
        
        self.device_replicate = device_replicate
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
        if self.mode == "UVA":
            edge_id = torch.zeros(1, dtype=torch.long)
            self.quiver = qv.new_quiver_from_csr_array(self.csr_topo.indptr, self.csr_topo.indices, edge_id, self.device, self.device_replicate)

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
        return self.csr_topo, self.sizes, self.mode, self.device_replicate
    
    @classmethod
    def lazy_from_ipc_handle(cls, ipc_handle):
        csr_topo, sizes, mode, device_replicate = ipc_handle
        return cls(csr_topo, sizes, -1, mode, device_replicate)
