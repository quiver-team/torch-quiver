import torch
import torch_quiver as qv

import random
import time
import numpy as np
import sys
import torch.multiprocessing as mp
import gc
import os.path as osp
from quiver.shard_tensor import ShardTensor as PyShardTensor
from quiver.shard_tensor import ShardTensorConfig
from quiver.async_feature import TorchShardTensor
from torch_geometric.utils import to_undirected, dropout_adj
from scipy.sparse import csr_matrix


# data_root = "/home/zy/papers/ogbn_papers100M/raw/"
# label = np.load(osp.join(data_root, "node-label.npz"))
# data = np.load(osp.join(data_root, "data.npz"))

def get_csr_from_coo(edge_index):
    src = edge_index[0]
    dst = edge_index[1]
    node_count = max(np.max(src), np.max(dst))
    data = np.zeros(dst.shape, dtype=np.int32)
    csr_mat = csr_matrix(
        (data, (edge_index[0], edge_index[1])))
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

    torch.save(indptr, "/home/zy/papers/ogbn_papers100M/csr/indptr.pt")
    torch.save(indices, "/home/zy/papers/ogbn_papers100M/csr/indices.pt")

def pyshard_tensor_ipc_child_proc(rank, ipc_handle, tensor):

    NUM_ELEMENT = 1000000
    SAMPLE_SIZE = 80000
    torch.cuda.set_device(rank)
    new_shard_tensor = PyShardTensor.new_from_share_ipc(ipc_handle, rank)

    host_indice = np.random.randint(0, 2 * NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)
    device_indices = indices.to(rank)

    
    ###############################
    # Calculate From New Tensor
    ###############################
    feature = new_shard_tensor[device_indices]
    print(f"{'#' * 40}")
    start = time.time()
    feature = new_shard_tensor[device_indices]
    consumed_time = time.time() - start
    feature = feature.cpu().numpy()
    feature_gt = tensor[indices].numpy()
    print("Correctness Check : ", np.array_equal(feature, feature_gt))


    print(
        f"TEST SUCCEED!, With Memory Bandwidth = {feature.size * 4 / consumed_time / 1024 / 1024 / 1024} GB/s, consumed {consumed_time}s")
    

def process_feature():
    print("LOG>>> Load Finished")
    NUM_ELEMENT = data["num_nodes_list"][0]

    nid_feat = data["node_feat"]
    tensor = torch.from_numpy(nid_feat).type(torch.float)
    print("LOG>>> Begin Process")
    torch.save(tensor, "/home/zy/papers/ogbn_papers100M/feat/feature.pt")

def process_label():
    print("LOG>>> Load Finished")
    node_label = label["node_label"]
    tensor = torch.from_numpy(node_label).type(torch.float)
    torch.save(tensor, "/home/zy/papers/ogbn_papers100M/label/label.pt")

def feature_test():
    SAMPLE_SIZE = 100000
    NUM_ELEMENT = 111059956
    
    host_indice = np.random.randint(0, NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)
    device_indices = indices.to(0)

    print("LOG>>> Begin Process")
    tensor = torch.load("/home/zy/papers/ogbn_papers100M/feat/sort_feature.pt")

    
    shard_tensor_config = ShardTensorConfig({})
    shard_tensor = PyShardTensor(0, shard_tensor_config)

    shard_tensor.from_cpu_tensor(tensor)
    


    '''
    # warm up
    res = shard_tensor[device_indices]


    start = time.time()
    feature = shard_tensor[device_indices]
    consumed_time = time.time() - start
    #feature = feature.cpu().numpy()

    #feature_gt = tensor[indices].numpy()
    
    #print("test complete")
    #print("Correctness Check : ", np.array_equal(feature, feature_gt))


    #print(
    #    f"TEST SUCCEED!, With Memory Bandwidth = {feature.size * 4 / consumed_time / 1024 / 1024 / 1024} GB/s, consumed {consumed_time}s")
    '''
    print("begin delete")
    shard_tensor.delete()
    print("after delete")

def sort_feature():
    NUM_ELEMENT = 111059956
    indptr = torch.load("/home/zy/papers/ogbn_papers100M/csr/indptr.pt")
    feature = torch.load("/home/zy/papers/ogbn_papers100M/feat/feature.pt")
    prev = torch.LongTensor(indptr[:-1])
    sub = torch.LongTensor(indptr[1:])
    deg = sub - prev
    sorted_deg, prev_order = torch.sort(deg, descending=True)
    total_num = NUM_ELEMENT
    total_range = torch.arange(total_num, dtype=torch.long)
    feature = feature[prev_order]
    torch.save(feature, "/home/zy/papers/ogbn_papers100M/feat/sort_feature.pt")
    torch.save(prev_order, "/home/zy/papers/ogbn_papers100M/feat/prev_order.pt")

# process_topo()
# qv.init_p2p()
# process_feature()
# process_label()
# feature_test()
sort_feature()
