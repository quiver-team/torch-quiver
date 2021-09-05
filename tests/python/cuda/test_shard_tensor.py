import torch
import torch_quiver as qv
import random
import time
import numpy as np
import sys
import torch.multiprocessing as mp
import gc  


def test_shard_tensor_item():
    item = qv.ShardTensorItem()
    item.device = 1
    item.shape = [1,2,3]
    item.mem_handle = "mem_handle"
    print(item.device, item.shape, item.mem_handle)
    print("ShardTensorItem TEST SUCCEED")
    
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

def child_proc(mem_handle0, device_0, shape0, mem_handle1, device_1, shape1):
    item0 = qv.ShardTensorItem()
    item0.mem_handle = mem_handle0
    item0.device = device_0
    item0.shape = shape0
    
    item1 = qv.ShardTensorItem()
    item1.mem_handle = mem_handle1
    item1.device = device_1
    item1.shape = shape1
    
    shard_tensor = qv.ShardTensor(1)
    shard_tensor.append(item0)
    shard_tensor.append(item1)
    
    
    

def test_shard_tensor_ipc():
    NUM_ELEMENT = 1000000
    SAMPLE_SIZE = 80000
    FEATURE_DIM = 600
    gc.disable()
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
    
    indices = torch.from_numpy(host_indice).type(torch.long)
    indices = indices.to("cuda:0")
    
    print(
        f"device_0_tensor device {device_0_tensor.device}\ndevice_1_tensor device {device_1_tensor.device}")
    shard_tensor2 = qv.ShardTensor(0)
    shard_tensor2.append(device_0_tensor, 0)
    shard_tensor2.append(device_1_tensor, 1)
    
    ipc_res = shard_tensor2.share_ipc()
    print(ipc_res[0].device, ipc_res[0].mem_handle)
    process = mp.Process(target=child_proc, args = (ipc_res[0].mem_handle, ipc_res[0].device, ipc_res[0].shape,
                                                    ipc_res[1].mem_handle, ipc_res[1].device, ipc_res[1].shape)
                        )
    process.start()
    process.join()
    gc.enable()
    
    
qv.init_p2p()
test_shard_tensor_item()
test_shard_tensor_intra_process()
test_shard_tensor_ipc()