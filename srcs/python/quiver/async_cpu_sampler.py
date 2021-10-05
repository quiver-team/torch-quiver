import copy
from typing import List, Optional, Tuple, NamedTuple, Union, Callable

import torch
from torch import Tensor
from torch_sparse import SparseTensor


class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)


def prepare_adj(edge_index):
    num_nodes = int(edge_index.max()) + 1
    value = None
    adj_t = SparseTensor(row=edge_index[0],
                         col=edge_index[1],
                         value=value,
                         sparse_sizes=(num_nodes, num_nodes)).t()
    adj_t.share_memory_()
    return adj_t


class AsyncNeighborSampler:
    def __init__(self, adj: Optional[SparseTensor] = None):
        self.adj = adj

    def sample_layer(self, batch, size):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)
        n_id = batch
        n_id, count = self.adj.sample_layer(n_id, size, replace=False)
        return n_id, count
