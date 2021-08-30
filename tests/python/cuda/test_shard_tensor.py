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
    indices = torch.randint(0, 2 * NUM_ELEMENT, (SAMPLE_SIZE, )).type(torch.long)
    indices = indices.to("cuda:0")
    
    # warm up
    feature = shard_tensor[indices]
    torch.cuda.synchronize()
    
    start = time.time()
    feature = shard_tensor[indices]
    torch.cuda.synchronize()
    print(f"gathered data shape = {feature.shape}, consumed {time.time() - start}")
    feature_sum = torch.sum(feature)
    
    indice_0_tensor = indices[indices < NUM_ELEMENT]
    indice_1_tensor = indices[indices >= NUM_ELEMENT]
    feature_0_tensor = device_0_tensor[indice_0_tensor]
    feature_1_tensor = device_1_tensor[indice_1_tensor].to("cuda:0")
    feature_sum_gt = torch.sum(feature_0_tensor) + torch.sum(feature_1_tensor)    
    
    assert torch.equal(feature_sum, feature_sum_gt), "feature collection check failed"
    
    print("TEST SUCCEED!")

test_shard_tensor_intra_process()
    
    
