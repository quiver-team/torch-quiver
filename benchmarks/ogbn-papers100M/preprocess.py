import torch
import torch_quiver as qv

import random
import time
import numpy as np
import sys
import torch.multiprocessing as mp
import os.path as osp
from numpy import genfromtxt
# from quiver.shard_tensor import ShardTensor as PyShardTensor
# from quiver.shard_tensor import ShardTensorConfig
# from torch_geometric.utils import to_undirected, dropout_adj
from scipy.sparse import csr_matrix


data_root = "/data/papers/ogbn_papers100M/raw/"
label = np.load(osp.join(data_root, "node-label.npz"))
data = np.load(osp.join(data_root, "data.npz"))

def get_csr_from_coo(edge_index, reverse=False):
    src = edge_index[0]
    dst = edge_index[1]
    if reverse:
        dst, src = src, dst
    node_count = max(np.max(src), np.max(dst))
    data = np.zeros(dst.shape, dtype=np.int32)
    csr_mat = csr_matrix(
        (data, (src, dst)))
    return csr_mat

def process_topo():
    edge_index = data["edge_index"]
    print("LOG>>> Load Finished")
    num_nodes = data["num_nodes_list"][0]

    print("LOG>>> Begin Process")

    # edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    csr_mat = get_csr_from_coo(edge_index)
    indptr = csr_mat.indptr
    indices = csr_mat.indices
    indptr = torch.from_numpy(indptr).type(torch.long)
    indices = torch.from_numpy(indices).type(torch.long)

    print("LOG>>> Begin Save")

    torch.save(indptr, "/data/papers/ogbn_papers100M/csr/indptr.pt")
    torch.save(indices, "/data/papers/ogbn_papers100M/csr/indices.pt")

    csr_mat = get_csr_from_coo(edge_index, True)
    indptr_reverse = csr_mat.indptr
    indices_reverse = csr_mat.indices
    indptr_reverse = torch.from_numpy(indptr_reverse).type(torch.long)
    indices_reverse = torch.from_numpy(indices_reverse).type(torch.long)
    
    torch.save(indptr_reverse, "/data/papers/ogbn_papers100M/csr/indptr_reverse.pt")
    torch.save(indices_reverse, "/data/papers/ogbn_papers100M/csr/indices_reverse.pt")

def process_feature():
    print("LOG>>> Load Finished")
    NUM_ELEMENT = data["num_nodes_list"][0]

    nid_feat = data["node_feat"]
    tensor = torch.from_numpy(nid_feat).type(torch.float)
    print("LOG>>> Begin Process")
    torch.save(tensor, "/data/papers/ogbn_papers100M/feat/feature.pt")

def process_label():
    print("LOG>>> Load Finished")
    node_label = label["node_label"]
    tensor = torch.from_numpy(node_label).type(torch.long)
    torch.save(tensor, "/data/papers/ogbn_papers100M/label/label.pt")

def sort_feature():
    NUM_ELEMENT = 111059956
    indptr = torch.load("/data/papers/ogbn_papers100M/csr/indptr_reverse.pt")
    feature = torch.load("/data/papers/ogbn_papers100M/feat/feature.pt")
    prev = torch.LongTensor(indptr[:-1])
    sub = torch.LongTensor(indptr[1:])
    deg = sub - prev
    sorted_deg, prev_order = torch.sort(deg, descending=True)
    total_num = NUM_ELEMENT
    total_range = torch.arange(total_num, dtype=torch.long)
    feature = feature[prev_order]
    torch.save(feature, "/data/papers/ogbn_papers100M/feat/sort_feature.pt")
    torch.save(prev_order, "/data/papers/ogbn_papers100M/feat/prev_order.pt")

def process_index():
    data = genfromtxt('/data/papers/ogbn_papers100M/split/time/train.csv', delimiter='\n')
    data = data.astype(np.long)
    data = torch.from_numpy(data)
    torch.save(data, "/data/papers/ogbn_papers100M/index/train_idx.pt")

process_topo()
process_feature()
process_label()
sort_feature()
process_index()
