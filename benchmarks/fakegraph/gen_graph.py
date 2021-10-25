from numpy.core.numeric import indices
import torch

import numpy as np
import os.path as osp
from torch_geometric.utils import to_undirected
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix

num_node = 10000000
num_feat = 1024

a, m = 3., 20.  # shape and mode
deg = np.random.pareto(a, num_node + 1) * m + 1
deg = deg.astype(np.int64)
deg[0] = 0
print(np.max(deg))
print(np.mean(deg))
num_edge = np.sum(deg)
indptr = np.cumsum(deg)
indices = np.random.randint(0, high=num_node, size=num_edge, dtype=np.int64)
data = np.zeros(num_edge, dtype=np.int64)
csc = csc_matrix((data, indices, indptr))
coo = csc.tocoo()
row = coo.row
col = coo.col
new_row = np.concatenate((row, col))
new_col = np.concatenate((col, row))
data = np.zeros(num_edge * 2, dtype=np.int64)
csr = csr_matrix((data, (new_row, new_col)))

indptr = torch.from_numpy(csr.indptr.astype(np.int64))
indices = torch.from_numpy(csr.indices.astype(np.int64))
prev = indptr[:-1]
sub = indptr[1:]
deg = sub - prev
sorted_deg, _ = torch.sort(deg, descending=True)
indptr = torch.zeros_like(indptr)
indptr[1:] = torch.cumsum(sorted_deg, 0)
indptr[0] = 0
print(torch.max(sorted_deg))
print(torch.mean(sorted_deg.float()))

feature = torch.rand(num_node, num_feat)