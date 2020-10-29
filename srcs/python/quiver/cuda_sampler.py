import copy
import os
import time
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch_quiver as qv


class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size)


class CudaNeighborSampler(torch.utils.data.DataLoader):
    def __init__(self,
                 edge_index: torch.Tensor,
                 sizes: List[int],
                 node_idx: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 **kwargs):

        N = int(edge_index.max() + 1) if num_nodes is None else num_nodes
        # print('building quiver')
        t0 = time.time()
        edge_id = torch.zeros(edge_index.size(1), dtype=torch.long)
        self.quiver = qv.new_quiver_from_edge_index(N, edge_index, edge_id)
        d = time.time() - t0
        # print('build quiver took %fms' % (d * 1000))

        if node_idx is None:
            node_idx = torch.arange(N)
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero().view(-1)

        self.sizes = sizes

        super(CudaNeighborSampler, self).__init__(node_idx.tolist(),
                                                  collate_fn=self.sample,
                                                  **kwargs)

    def sample(self, batch):
        t0 = time.time()
        ret = self._sample(batch)
        d = time.time() - t0
        # print('sample took %fms' % (d * 1000))
        return ret

    def _sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        adjs: List[Adj] = []

        n_id = batch
        for size in self.sizes:
            result, row_idx, col_idx = self.quiver.sample_sub(n_id, size)
            assert (row_idx.max() < n_id.size(0))

            row_idx, col_idx = col_idx, row_idx
            edge_index = torch.stack([row_idx, col_idx], dim=0)

            size = torch.LongTensor([
                result.size(0),
                n_id.size(0),
            ])
            assert (row_idx.max() < result.size(0))
            assert (col_idx.max() < n_id.size(0))

            # print('size: %s' % (size))
            # FIXME: also sample e_id
            e_id = torch.tensor([])  # e_id is not used in the example
            adjs.append(Adj(edge_index, e_id, size))
            n_id = result

        if len(adjs) > 1:
            return batch_size, n_id, adjs[::-1]
        else:
            return batch_size, n_id, adjs[0]

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)
