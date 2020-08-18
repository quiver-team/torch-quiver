#!/usr/bin/env python3

import copy
import os
import os.path as osp
import time
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch_quiver as qv
from ogb.nodeproppred import PygNodePropPredDataset

from ogbn_products_sage.cuda_sampler import CudaNeighborSampler


def info(t, name=None):
    msg = ''
    if name:
        msg += name
    msg += ' ' + str(t.type())
    msg += ' ' + str(t.shape)
    print(msg)


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

    for idx, (batch_size, n_id, adjs) in enumerate(sampler):
        print('#%d' % (idx))


main()
