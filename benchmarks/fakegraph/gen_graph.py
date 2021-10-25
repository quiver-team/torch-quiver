from numpy.core.numeric import indices
import torch

import numpy as np
import os
import os.path as osp
from torch_geometric.utils import to_undirected
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix

os.mkdir('fake')
os.mkdir('fake/csr')
os.mkdir('fake/feat')
os.mkdir('fake/index')
os.mkdir('fake/label')
csr_root = osp.join('./fake', 'csr')
label_root = osp.join('./fake', 'label')
feat_root = osp.join('./fake', 'feat')
index_root = osp.join('./fake', 'index')

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
label = torch.randint(0, high=100, size=(num_node,1))
train_idx = torch.randperm(num_node)
torch.save(feature, osp.join(feat_root, 'feat.pt'))
torch.save(label, osp.join(label_root, 'label.pt'))
torch.save(train_idx, osp.join(index_root, 'index.pt'))
torch.save(indptr, osp.join(csr_root, 'indptr.pt'))
torch.save(indices, osp.join(csr_root, 'indices.pt'))