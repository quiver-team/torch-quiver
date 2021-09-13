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



def test_shard_tensor_item():
   
    NUM_ELEMENT = 100
    FEATURE_DIM = 60
    #########################
    # Init With Numpy
    ########################
    host_tensor = np.random.randint(
        0, high=10, size=(2 * NUM_ELEMENT, FEATURE_DIM))

    device_0_tensor = torch.from_numpy(
        host_tensor[: NUM_ELEMENT]).type(torch.float32)
    device_1_tensor = torch.from_numpy(
        host_tensor[NUM_ELEMENT:]).type(torch.float32)

    shard_tensor = qv.ShardTensor(0)
    shard_tensor.append(device_0_tensor, 0)
    shard_tensor.append(device_1_tensor, 1)
    
    res = shard_tensor.share_ipc()
    item = res[0].share_ipc()
    print(item[0], item[1], item[2])
    new_shard_tensor_item = qv.ShardTensorItem()
    new_shard_tensor_item.from_ipc(item)
    item1 = new_shard_tensor_item.share_ipc()
    assert item[1] == item1[1]
    
def test_shard_tensor_intra_process():
    NUM_ELEMENT = 1000000
    SAMPLE_SIZE = 80000
    FEATURE_DIM = 600
    #########################
    # Init With Numpy
    ########################
    host_tensor = np.random.randint(
        0, high=10, size=(2 * NUM_ELEMENT, FEATURE_DIM))
    host_indice = np.random.randint(0, 2 * NUM_ELEMENT - 1, (SAMPLE_SIZE, ))

    device_0_tensor = torch.from_numpy(
        host_tensor[: NUM_ELEMENT]).type(torch.float32)
    device_1_tensor = torch.from_numpy(
        host_tensor[NUM_ELEMENT:]).type(torch.float32)

    print(
        f"device_0_tensor device {device_0_tensor.device}\ndevice_1_tensor device {device_1_tensor.device}")
    shard_tensor = qv.ShardTensor(0)
    shard_tensor.append(device_0_tensor, 0)
    shard_tensor.append(device_1_tensor, 1)
    print("shard_tensor shape = ", shard_tensor.shape())

    indices = torch.from_numpy(host_indice).type(torch.long)
    indices = indices.to("cuda:0")

    # warm up
    feature = shard_tensor[indices]
    torch.cuda.synchronize()

    start = time.time()
    feature = shard_tensor[indices]
    torch.cuda.synchronize()
    consumed_time = time.time() - start
    print(
        f"gathered data shape = {feature.shape}, consumed {time.time() - start}")
    torch.cuda.synchronize()
    whole_tensor = torch.from_numpy(
        host_tensor).type(torch.float32).to("cuda:0")
    start = time.time()
    res = whole_tensor[indices]
    torch.cuda.synchronize()
    print(
        f"gathered data shape using torch tensor = {res.shape}, consumed {time.time() - start}")

    feature = feature.cpu().numpy()
    feature_gt = host_tensor[host_indice]
    assert np.array_equal(feature, feature_gt), "TEST FAILED"
    print(
        f"TEST SUCCEED!, With Memory Bandwidth = {feature.size * 4 / consumed_time / 1024 / 1024 / 1024} GB/s")

def child_proc(ipc_item0, ipc_item1):
    current_device = 3
    torch.cuda.set_device(current_device)
    NUM_ELEMENT = 10000
    SAMPLE_SIZE = 800
    host_indice = np.random.randint(0,1 * NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices_part1 = torch.from_numpy(host_indice).type(torch.long)
    indices_part1 = indices_part1.to(current_device)
    
    host_indice_2 = np.random.randint(1 * NUM_ELEMENT, 2* NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices_part2 = torch.from_numpy(host_indice_2).type(torch.long)
    indices_part2 = indices_part2.to(current_device)
    
    host_indice3 = np.random.randint(0,2 * NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice3).type(torch.long)
    indices = indices.to(current_device)
    
    
    item0 = qv.ShardTensorItem()
    item0.from_ipc(ipc_item0)
    
    item1 = qv.ShardTensorItem()
    item1.from_ipc(ipc_item1)
    
    shard_tensor = qv.ShardTensor(current_device)
    shard_tensor.append(item0)
    shard_tensor.append(item1)
    
    print(f"check shard tensor shape ", shard_tensor.shape())
    start = time.time()
    feature = shard_tensor[indices_part1]
    torch.cuda.synchronize()
    print(
        f"gathered upper half data shape = {feature.shape}, consumed {time.time() - start}")
    
    print(f"check shard tensor shape ", shard_tensor.shape())
    start = time.time()
    feature = shard_tensor[indices_part2]
    torch.cuda.synchronize()
    print(
        f"gathered down half data shape = {feature.shape}, consumed {time.time() - start}")
    
    print(f"check shard tensor shape ", shard_tensor.shape())
    start = time.time()
    feature = shard_tensor[indices]
    torch.cuda.synchronize()
    print(
        f"gathered whole data shape = {feature.shape}, consumed {time.time() - start}")
    
    
    

def test_shard_tensor_ipc():
    NUM_ELEMENT = 10000
    SAMPLE_SIZE = 800
    FEATURE_DIM = 600
    gc.disable()
    #########################
    # Init With Numpy
    ########################
    torch.cuda.set_device(1)

    host_tensor = np.random.randint(
        0, high=10, size=(2 * NUM_ELEMENT, FEATURE_DIM))

    device_0_tensor = torch.from_numpy(
        host_tensor[: NUM_ELEMENT]).type(torch.float32)
    device_1_tensor = torch.from_numpy(
        host_tensor[NUM_ELEMENT:]).type(torch.float32)
    
    print(
        f"device_0_tensor device {device_0_tensor.device}\ndevice_1_tensor device {device_1_tensor.device}")
    shard_tensor2 = qv.ShardTensor(1)
    shard_tensor2.append(device_0_tensor, 2)
    shard_tensor2.append(device_1_tensor, 3)


    ipc_res = shard_tensor2.share_ipc()
    print(ipc_res[0].share_ipc()[1] == ipc_res[1].share_ipc()[1])
    process = mp.Process(target=child_proc, args = (ipc_res[0].share_ipc(),ipc_res[1].share_ipc()))
    process.start()
    process.join()
    gc.enable()
    

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
    host_indice = np.random.randint(0,2 * NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)
    indices = indices.to("cuda:0")
    shard_tensor_config = ShardTensorConfig({0: '1.1G', 1: "900M"})
    shard_tensor = PyShardTensor(0, shard_tensor_config)
    shard_tensor.from_cpu_tensor(tensor)
    start = time.time()
    feature = shard_tensor[indices]
    torch.cuda.synchronize()
    consumed_time = time.time() - start
    feature = feature.cpu().numpy()
    print(
        f"TEST SUCCEED!, With Memory Bandwidth = {feature.size * 4 / consumed_time / 1024 / 1024 / 1024} GB/s")

    

if __name__ == "__main__":
    mp.set_start_method("spawn")
    qv.init_p2p()
    #test_shard_tensor_item()
    #test_shard_tensor_intra_process()
    test_shard_tensor_ipc()
    # test_py_shard_tensor_basic()
