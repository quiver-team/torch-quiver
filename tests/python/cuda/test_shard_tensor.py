import torch
import torch_quiver as qv
import random
import time

def test_shard_tensor_intra_process():
    device_0_tensor = torch.ones((1000000, 600), device = "cuda:0", dtype = torch.float32)
    device_1_tensor = torch.ones((1000000, 600), device = "cuda:1", dtype=torch.float32)
    print(f"device_0_tensor device {device_0_tensor.device}\ndevice_1_tensor device {device_1_tensor.device}")
    shard_tensor = qv.ShardTensor(0)
    shard_tensor.append(device_0_tensor)
    shard_tensor.append(device_1_tensor)
    print("shard_tensor shape = ", shard_tensor.shape())
    indices = torch.randint(0, 2000000, (800000, )).type(torch.long)
    indices = indices.to("cuda:0")
    start = time.time()
    feature = shard_tensor[indices]
    print(feature[0]);
    print(f"gathered data shape = {feature.shape}, consumed {time.time() - start}")
    

test_shard_tensor_intra_process()
    
    