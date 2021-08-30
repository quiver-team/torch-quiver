import torch
import torch_quiver as qv
import random
import time

def test_shard_tensor_intra_process():
    NUM_ELEMENT = 1000000
    SAMPLE_SIZE = 80000
    FEATURE_DIM = 600
    device_0_tensor = torch.randint(0, 10, (NUM_ELEMENT, FEATURE_DIM), device = "cuda:0", dtype = torch.float32)
    device_1_tensor = torch.randint(0, 10, (NUM_ELEMENT, FEATURE_DIM), device = "cuda:1", dtype=torch.float32)
    print(f"device_0_tensor device {device_0_tensor.device}\ndevice_1_tensor device {device_1_tensor.device}")
    shard_tensor = qv.ShardTensor(0)
    shard_tensor.append(device_0_tensor)
    shard_tensor.append(device_1_tensor)
    print("shard_tensor shape = ", shard_tensor.shape())
    indices = torch.randint(0, 2 * NUM_ELEMENT - 1, (SAMPLE_SIZE, )).type(torch.long)
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
    
    print("TEST SUCCEED!")

test_shard_tensor_intra_process()
    
    
