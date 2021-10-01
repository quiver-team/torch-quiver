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



data_root = "/home/zy/papers/ogbn_papers100M/raw/"
data = np.load(osp.join(data_root, "data.npz"))

def process_topo():
    edge_index = data["edge_index"]
    print("LOG>>> Load Finished")
    edge_index = torch.from_numpy(edge_index)
    num_nodes = data["num_nodes_list"][0]

    print("LOG>>> Begin Process")

    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    
    print("LOG>>> Begin Save")



    np.save("paper100m_undirected", edge_index)

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
    NUM_ELEMENT = data["num_nodes_list"][0]
    SAMPLE_SIZE = 80000
    
    host_indice = np.random.randint(0, NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)
    device_indices = indices.to(0)

    nid_feat = data["node_feat"]
    tensor = torch.from_numpy(nid_feat)
    shard_tensor_config = ShardTensorConfig({0:"20G", 1:"20G"})
    shard_tensor = PyShardTensor(0, shard_tensor_config)

    shard_tensor.from_cpu_tensor(tensor)


    # warm up
    res = shard_tensor[device_indices]


    start = time.time()
    feature = new_shard_tensor[device_indices]
    consumed_time = time.time() - start
    feature = feature.cpu().numpy()
    feature_gt = tensor[indices].numpy()
    print("Correctness Check : ", np.array_equal(feature, feature_gt))


    print(
        f"TEST SUCCEED!, With Memory Bandwidth = {feature.size * 4 / consumed_time / 1024 / 1024 / 1024} GB/s, consumed {consumed_time}s")
    


#process_topo()
qv.init_p2p()
process_feature()

