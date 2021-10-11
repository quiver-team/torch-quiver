import asyncio
import concurrent
import copy
import os
import time
from typing import List, NamedTuple, Optional, Tuple
import numpy as np

import torch
from torch_sparse import SparseTensor
import torch_quiver as qv


class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size)


class AsyncCudaNeighborSampler:
    def __init__(self,
                 edge_index: Optional[torch.Tensor] = None,
                 csr_indptr: Optional[torch.Tensor] = None,
                 csr_indices: Optional[torch.Tensor] = None,
                 copy: Optional[bool] = False,
                 device: int = 0,
                 num_nodes: Optional[int] = None):

        # Initilization With COO
        if edge_index is not None:
            N = int(edge_index.max() + 1) if num_nodes is None else num_nodes
            edge_id = torch.zeros(1, dtype=torch.long)
            self.quiver = qv.new_quiver_from_edge_index(
                N, edge_index, edge_id, device)
        # Initialization With CSR
        if csr_indptr is not None and csr_indices is not None:
            print("LOG>>> Create Quiver From CSR")
            edge_id = torch.zeros(1, dtype=torch.long)
            self.quiver = qv.new_quiver_from_csr_array(csr_indptr, csr_indices,
                                                       edge_id, device, copy)

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
