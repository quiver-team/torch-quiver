import torch
import torch_quiver as qv
import random
import time
import numpy as np 

def test_shard_tensor_intra_process():
    NUM_ELEMENT = 1000000
    SAMPLE_SIZE = 80000
    FEATURE_DIM = 600
    #########################
    # Init With Numpy
    ########################
    host_tensor = np.random.randint(0, high=10, size=(2 * NUM_ELEMENT, FEATURE_DIM), dtype=np.float32)
    host_indice = np.random.randint(0, 2 * NUM_ELEMENT - 1, (SAMPLE_SIZE, ), dtype=np.int64)
    
    device_0_tensor = torch.from_numpy(host_tensor[: NUM_ELEMENT]).type(torch.float32).to("cuda:0")
    device_1_tensor =  torch.from_numpy(host_tensor[NUM_ELEMENT:]).type(torch.float32).to("cuda:1")
    
    print(f"device_0_tensor device {device_0_tensor.device}\ndevice_1_tensor device {device_1_tensor.device}")
    shard_tensor = qv.ShardTensor(0)
    shard_tensor.append(device_0_tensor)
    shard_tensor.append(device_1_tensor)
    print("shard_tensor shape = ", shard_tensor.shape())
    
    indices = torch.from_numpy(host_indice).type(torch.long)
    indices = indices.to("cuda:0")
    
    # warm up
    feature = shard_tensor[indices]
    torch.cuda.synchronize()
    
    start = time.time()
    feature = shard_tensor[indices]
    torch.cuda.synchronize()
    print(f"gathered data shape = {feature.shape}, consumed {time.time() - start}")
    feature_sum = torch.sum(feature)
    print(feature_sum)
    sum_gt = np.sum(host_tensor[host_indice])
    print(sum_gt)
    
    print("TEST SUCCEED!")

test_shard_tensor_intra_process()
    
    
