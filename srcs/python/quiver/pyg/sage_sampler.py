import torch
from torch import Tensor
from torch_sparse import SparseTensor

import torch_quiver as qv

from typing import List, Optional, Tuple, NamedTuple, Union, Callable

__all__ = ["GraphSageSampler", "GraphStructure"]

class GraphStructure:
    def __init__(self, edge_index=None, indptr=None, indices=None, eid=None):
        self.edge_index = edge_index
        self.indptr = indptr
        self.indices = indices
        self.eid = eid



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
    edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
            :obj:`torch_sparse.SparseTensor` that defines the underlying graph
            connectivity/message passing flow.
            :obj:`edge_index` holds the indices of a (sparse) symmetric
            adjacency matrix.
            If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its shape
            must be defined as :obj:`[2, num_edges]`, where messages from nodes
            :obj:`edge_index[0]` are sent to nodes in :obj:`edge_index[1]`
            (in case :obj:`flow="source_to_target"`).
            If :obj:`edge_index` is of type :obj:`torch_sparse.SparseTensor`,
            its sparse indices :obj:`(row, col)` should relate to
            :obj:`row = edge_index[1]` and :obj:`col = edge_index[0]`.
            The major difference between both formats is that we need to input
            the *transposed* sparse adjacency matrix.
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

    def __init__(self, edge_index: Tensor, sizes: List[int], device, num_nodes: Optional[int] = None, mode="UVA", device_replicate=True):
        edge_index = edge_index.to("cpu")
        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)

        self.sizes = sizes
        
        self.quiver = None
        # Obtain a *transposed* `SparseTensor` instance.
        if not self.is_sparse_tensor:
            if num_nodes is None:
                num_nodes = int(edge_index.max()) + 1

            value = None
            self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=value, sparse_sizes=(num_nodes, num_nodes)).t()
        else:
            self.adj_t = edge_index

        self.mode = mode
        if self.mode == "UVA":
            indptr, indices, _ = self.adj_t.csr()
            edge_id = torch.zeros(1, dtype=torch.long)

            self.quiver = qv.new_quiver_from_csr_array(indptr, indices, edge_id, device, device_replicate)
            if not device_replicate:
                # Save to prevent gc
                self.indptr = indptr
                self.indices = indices
            
        else:
            pass
        
        self.device_replicate = device_replicate
        self.device = device

    
    def sample_layer(self, batch, size):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)
        n_id = batch.to(torch.device(self.device))
        n_id, count = self.quiver.sample_neighbor(0, n_id, size)
        return n_id, count
    
    def reindex(self, inputs, outputs, counts):
        return qv.reindex_single(inputs, outputs, counts)

    def sample(self, input_nodes):
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

