#!/usr/bin/env python3

import copy
import os
import os.path as osp
import time
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch_quiver as qv
from ogb.nodeproppred import PygNodePropPredDataset


def info(t, name=None):
    msg = ''
    if name:
        msg += name
    msg += ' ' + str(t.type())
    msg += ' ' + str(t.shape)
    print(msg)


class CudaNeighborSampler(torch.utils.data.DataLoader):
    def __init__(self,
                 edge_index: torch.Tensor,
                 sizes: List[int],
                 node_idx: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 **kwargs):

        N = int(edge_index.max() + 1) if num_nodes is None else num_nodes
        print('building quiver')
        t0 = time.time()
        self.quiver = qv.new_quiver_from_edge_index(N, edge_index)
        d = time.time() - t0
        print('build quiver took %fms' % (d * 1000))

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
        print('sample took %fms' % (d * 1000))
        return ret

    def _sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)
        print('sample batch size %d' % (batch_size))

        # adjs: List[Adj] = []

        n_id = batch
        for size in self.sizes:
            result, row_idx, col_idx = self.quiver.sample_sub(n_id, size)
            info(result)
            n_id = result

            edge_index = torch.stack([row_idx, col_idx], dim=0)
            # adjs.append(Adj(edge_index, e_id, size))

        return batch_size, batch
        # if len(adjs) > 1:
        #     return batch_size, n_id, adjs[::-1]
        # else:
        #     return batch_size, n_id, adjs[0]

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)


def load_data():
    home = os.getenv('HOME')
    data_dir = os.path.join(home, '.pyg')
    root = os.path.join(data_dir, 'data', 'products')
    filename = os.path.join(
        root, 'ogbn_products_pyg/processed/geometric_data_processed.pt')
    data, _ = torch.load(filename)
    return data.edge_index


def load_dataset():
    home = os.getenv('HOME')
    data_dir = osp.join(home, '.pyg')
    root = osp.join(data_dir, 'data', 'products')
    dataset = PygNodePropPredDataset('ogbn-products', root)
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    return data, split_idx


def main():
    data, split_idx = load_dataset()
    train_idx = split_idx['train']
    sampler = CudaNeighborSampler(
        data.edge_index,
        node_idx=train_idx,
        sizes=[15, 10, 5],
        batch_size=1024,
    )

    # for idx, (batch_size, n_id, adjs) in enumerate(train_loader):
    for idx, (batch_size, batch) in enumerate(sampler):
        print('#%d' % (idx))


main()
