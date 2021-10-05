import os
import os.path as osp

import torch
import numpy as np
from scipy.sparse import csr_matrix

import time
from quiver.shard_tensor import ShardTensorConfig

import random
from typing import List, NamedTuple, Optional, Tuple
"""
We Balance Data Access On CPU by using numa interleave memory allocation
We Balance Data Access On GPU By Reindex
"""


def reindex_by_config(adj_csr, graph_feature, gpu_portion):
    degree = adj_csr.indptr[1:] - adj_csr.indptr[:-1]
    node_count = degree.shape[0]
    # first reorder to ensure hottest data is on gpu
    reordered_index = torch.argsort(degree, dim=0)

    # shuffle gpu data to pursue accessing balance
    random.shuffle(reordered_index[:int(node_count * gpu_portion)])

    # reorder graph and feature
    # 1. reorder feature
    graph_feature = graph_feature[reordered_index]

    # 2. reoder graph
