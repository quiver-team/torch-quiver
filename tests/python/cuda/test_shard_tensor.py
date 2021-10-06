import torch
import torch_quiver as qv

import random
import time
import numpy as np
import sys
import torch.multiprocessing as mp
import gc

from quiver.shard_tensor import ShardTensor as PyShardTensor
from quiver.shard_tensor import ShardTensorConfig
from quiver.async_feature import TorchShardTensor

def test_normal_feature_collection():
    NUM_ELEMENT = 1000000
    SAMPLE_SIZE = 80000
    FEATURE_DIM = 600
    gc.disable()
    #########################
    # Init With Numpy
    ########################
    torch.cuda.set_device(0)

    host_tensor = np.random.randint(
        0, high=10, size=(2 * NUM_ELEMENT, FEATURE_DIM))
    tensor = torch.from_numpy(host_tensor).type(torch.float32)
    host_indice = np.random.randint(0, 2 * NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)
    # warm up
    res = tensor[indices].to(0)

    start = time.time()
    feature = tensor[indices]
    feature = feature.to(0)
    consumed_time = time.time() - start
    feature = feature.cpu().numpy()
    print(
        f"TEST SUCCEED!, With Memory Bandwidth = {feature.size * 4 / consumed_time / 1024 / 1024 / 1024} GB/s")


def test_py_shard_tensor_basic():
    NUM_ELEMENT = 1000000
    SAMPLE_SIZE = 80000
    FEATURE_DIM = 600
    gc.disable()
    #########################
    # Init With Numpy
    ########################
    torch.cuda.set_device(0)

    host_tensor = np.random.randint(
        0, high=10, size=(2 * NUM_ELEMENT, FEATURE_DIM))
    tensor = torch.from_numpy(host_tensor).type(torch.float32)
    host_indice = np.random.randint(0, 2 * NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)
    indices = indices.to(0)
    shard_tensor_config = ShardTensorConfig({1:"200M"})
    shard_tensor = PyShardTensor(0, shard_tensor_config)
    shard_tensor.from_cpu_tensor(tensor)

    

    start = time.time()
    feature = shard_tensor[indices]
    consumed_time = time.time() - start
    feature = feature.cpu().numpy()
    feature_gt = host_tensor[host_indice]
    assert np.array_equal(feature_gt, feature)
    print(
        f"TEST SUCCEED!, With Memory Bandwidth = {feature.size * 4 / consumed_time / 1024 / 1024 / 1024} GB/s")

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
    print(feature[0][:10])
    print(feature_gt[0][:10])
    print("Correctness Check : ", np.array_equal(feature, feature_gt))


    print(
        f"TEST SUCCEED!, With Memory Bandwidth = {feature.size * 4 / consumed_time / 1024 / 1024 / 1024} GB/s, consumed {consumed_time}s")
    
def test_py_shard_tensor_ipc():
    NUM_ELEMENT = 1000000
    FEATURE_DIM = 600
    gc.disable()
    #########################
    # Init With Numpy
    ########################
    current_device = 1
    torch.cuda.set_device(current_device)

    host_tensor = np.random.randint(
        0, high=10, size=(2 * NUM_ELEMENT, FEATURE_DIM))
    tensor = torch.from_numpy(host_tensor).type(torch.float32)
    tensor.share_memory_()
    # 所有数据均在CPU
    #shard_tensor_config = ShardTensorConfig({})
    # 20%数据在GPU0，80%的数据在CPU
    # 20%的数据在GPU0， 20%的数据在GPU3，60%的数据在CPU
    shard_tensor_config = ShardTensorConfig({0:"0.9G", 1:"0.9G"})

    shard_tensor = PyShardTensor(current_device, shard_tensor_config)
    shard_tensor.from_cpu_tensor(tensor)
    #shard_tensor.append(tensor[:400000], 0)
    ##shard_tensor.append(tensor[400000:800000], 1)
    #shard_tensor.append(tensor[800000:1200000], 2)
    #shard_tensor.append(tensor[1200000:], -1)



    ##########################
    # Create IPC Handle
    #########################
    ipc_handle = shard_tensor.share_ipc()
    process = mp.Process(target=pyshard_tensor_ipc_child_proc, args=(0, ipc_handle, tensor))
    process.start()
    process.join()


def torch_child_proc(rank, ws, cpu_tensor, gpu_tensors, range_list, indices):
    shard_tensor = TorchShardTensor(
        rank, ws, cpu_tensor, gpu_tensors, range_list)
    feature = shard_tensor.collect(indices)
    torch.cuda.synchronize(0)
    start = time.time()
    feature = shard_tensor.collect(indices)
    torch.cuda.synchronize(0)
    consumed_time = time.time() - start
    feature = feature.cpu().numpy()
    print(
        f"TEST SUCCEED!, With Memory Bandwidth = {feature.size * 4 / consumed_time / 1024 / 1024 / 1024} GB/s")

def test_products():

    from ogb.nodeproppred import PygNodePropPredDataset
    root = "/home/dalong/data/products/"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    data = dataset[0]

    NUM_ELEMENT = 1000000
    FEATURE_DIM = 600

    torch.cuda.set_device(0)

    host_tensor = np.random.rand(2 * NUM_ELEMENT, FEATURE_DIM)
    tensor = torch.from_numpy(host_tensor).type(torch.float32)

    tensor = torch.Tensor([[ 0.0319, -0.1959,  0.0520, -0.0633, -0.2299, -0.0221,  0.4046, -0.1079,
         0.0326,  0.0603]])

    # Uncomment this to see wierd behavior of shard_tensor
    # data.x = tensor

    feature = data.x
    print(f"original data ", feature[0][:10])

    device_feature = feature.to(0)

    shard_tensor_config = ShardTensorConfig({0:"200M"})
    shard_tensor = PyShardTensor(0, shard_tensor_config)

    shard_tensor.from_cpu_tensor(feature)

    print("shard_tensor.shape", shard_tensor.size(0))

    nid = torch.LongTensor([0]).to(0)

    print(shard_tensor[nid][0][:10])
    print(device_feature[nid][0][:10])







def test_torch_shard_tensor():
    NUM_ELEMENT = 1000000
    SAMPLE_SIZE = 80000
    FEATURE_DIM = 600
    #########################
    # Init With Numpy
    ########################
    torch.cuda.set_device(0)

    host_tensor = np.random.randint(
        0, high=10, size=(NUM_ELEMENT, FEATURE_DIM))
    tensor = torch.from_numpy(host_tensor).type(torch.float32)
  
    host_indice = np.random.randint(0, NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)
    indices = indices.to("cuda:0")
    range_list = [0, NUM_ELEMENT // 5, 2 * NUM_ELEMENT // 5,
                  3 * NUM_ELEMENT // 5, 4 * NUM_ELEMENT // 5, NUM_ELEMENT]
    gpu_tensors = []
    for rank in range(4):
        beg = range_list[rank]
        end = range_list[rank + 1]
        t = tensor[beg:end].clone()
        if 0 != rank:
            t = t.to(rank)
        gpu_tensors.append(t)
    cpu_beg = range_list[4]
    cpu_end = NUM_ELEMENT
    cpu_tensor = tensor[cpu_beg: cpu_end].clone()
    proc = mp.Process(target=torch_child_proc, args=(
        0, 4, cpu_tensor, gpu_tensors, range_list, indices))
    proc.start()
    proc.join()

def basic_test():
    NUM_ELEMENT = 1000000
    SAMPLE_SIZE = 80000
    FEATURE_DIM = 600
    #########################
    # Init With Numpy
    ########################

    src_device = 0
    dst_device = 2

    torch.cuda.set_device(src_device)

    host_tensor = np.random.randint(
        0, high=10, size=(NUM_ELEMENT, FEATURE_DIM))
    tensor = torch.from_numpy(host_tensor).type(torch.float32)
  
    host_indice = np.random.randint(0, NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)

    indices = indices.to(src_device)
    tensor = tensor.to(dst_device)

    # warm up
    data = tensor[indices.to(dst_device, non_blocking=True)].to(src_device)
    torch.cuda.synchronize(src_device)


    # test 
    start = time.time()
    data = tensor[indices.to(dst_device, non_blocking=True)].to(src_device)
    torch.cuda.synchronize(src_device)
    consumed_time = time.time() - start

    feature = data.cpu().numpy()
    print(
        f"TEST SUCCEED!, With Memory Bandwidth = {feature.size * 4 / consumed_time / 1024 / 1024 / 1024} GB/s")


def test_feature_collection_gpu_utlization():
    NUM_ELEMENT = 1000000
    FEATURE_DIM = 600
    SAMPLE_SIZE = 80000

    #########################
    # Init With Numpy
    ########################
    current_device = 0
    torch.cuda.set_device(current_device)

    host_tensor = np.random.randint(
        0, high=10, size=(2 * NUM_ELEMENT, FEATURE_DIM))
    tensor = torch.from_numpy(host_tensor).type(torch.float32)
    shard_tensor_config = ShardTensorConfig({0:"0.9G", 1:"0.9G", 2:"0.9G", 3:"0.9G"})
    shard_tensor = PyShardTensor(current_device, shard_tensor_config)
    shard_tensor.from_cpu_tensor(tensor)
    host_indice = np.random.randint(0, 2 * NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)
    device_indices = indices.to(current_device)

    while True:
        res = shard_tensor[device_indices]
    

def test_delete():

    NUM_ELEMENT = 20000000
    FEATURE_DIM = 128

    #########################
    # Init With Numpy
    ########################
    current_device = 0
    torch.cuda.set_device(current_device)

    host_tensor = np.random.randint(
        0, high=10, size=(2 * NUM_ELEMENT, FEATURE_DIM))
    tensor = torch.from_numpy(host_tensor).type(torch.float32)
    shard_tensor_config = ShardTensorConfig({})
    shard_tensor = PyShardTensor(current_device, shard_tensor_config)
    shard_tensor.from_cpu_tensor(tensor)

    print(f"check delete")
    beg = time.time()
    shard_tensor.delete()
    end = time.time()
    print(f"papers100M took {end - beg}")
    print("test complete")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    qv.init_p2p()
    #test_delete()
    #test_py_shard_tensor_basic()
    #test_normal_feature_collection()
    #basic_test()
    test_products()
    #test_py_shard_tensor_ipc()
    #test_torch_shard_tensor()
    # test_py_shard_tensor_basic()
    #test_feature_collection_gpu_utlization()
