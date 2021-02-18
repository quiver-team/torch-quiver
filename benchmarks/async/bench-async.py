#!/usr/bin/env python3

import os
import os.path as osp
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from quiver.cuda_sampler import CudaNeighborSampler
from quiver.cuda_loader import CudaNeighborLoader
from quiver.profile_utils import StopWatch
import torch
import multiprocessing as mp


def async_sampler(data, train_idx, w):
    train_loader = CudaNeighborSampler(data.edge_index,
                                       node_idx=train_idx,
                                       mode='sync',
                                       sizes=[15, 10, 5],
                                       batch_size=1024,
                                       shuffle=True)
    w.tick('create train_loader')

    for i in range(10):
        for batch_size, n_id, adjs in train_loader:
            pass
        w.tick('one round')
    w.tick('sample')


def async_loader(data, train_idx, w):
    train_loader = CudaNeighborLoader(
        (data.edge_index, [15, 10, 5], train_idx), 1024, 1)
    w.tick('create train_loader')
    count = 0
    for i in range(10):
        for batch_size, n_id, adjs in train_loader:
            count += 1
            if count == 1:
                w.tick('init generator')
        w.tick('one round')
        if i == 9:
            train_loader.close()
            break
        train_loader.reset()
        w.tick('reset')
    w.tick('sample')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    w = StopWatch('main')
    home = os.getenv('HOME')
    data_dir = osp.join(home, '.pyg')
    root = osp.join(data_dir, 'data', 'products')
    dataset = PygNodePropPredDataset('ogbn-products', root)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-products')
    data = dataset[0]

    train_idx = split_idx['train']

    w.tick('load data')

    async_loader(data, train_idx, w)

    del w
