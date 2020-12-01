#!/usr/bin/env python3

import os
import os.path as osp
from ogb.nodeproppred import Evaluator, NodePropPredDataset
from quiver.cuda_sampler import CudaNeighborSampler
from quiver.cuda_loader import CudaNeighborLoader
from quiver.profile_utils import StopWatch
from pycuda import autoinit
import torch
import multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn')
    w = StopWatch('main')
    home = os.getenv('HOME')
    data_dir = osp.join(home, '.pyg')
    root = osp.join(data_dir, 'data', 'products')
    dataset = NodePropPredDataset('ogbn-products', root)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-products')
    data = dataset[0]

    train_idx = split_idx['train']

    w.tick('load data')
    graph, labels = data
# train_loader = CudaNeighborSampler(torch.LongTensor(graph['edge_index']),
#                                    node_idx=train_idx,
#                                    mode='coro',
#                                    sizes=[15, 10, 5],
#                                    batch_size=1024,
#                                    shuffle=True)
# w.tick('create train_loader')

# for batch_size, n_id, adjs in train_loader:
#     pass
# w.tick('sample')

    train_loader = CudaNeighborLoader((torch.LongTensor(graph['edge_index']),
                                    [15, 10, 5], train_idx),
                                    1024, 4)
    w.tick('create train_loader')

    for batch_size, n_id, adjs in train_loader:
        pass
    w.tick('sample')

    del w
