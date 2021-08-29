import torch
import torch_quiver as qv

def test_shard_tensor_intra_process():
    device_0_tensor = torch.randint(0, 100, (100, 30), device = "cuda:0", dtype = torch.float32)
    device_1_tensor = torch.randint(0, 100, (50, 30), device = "cuda:1", dtype=torch.float32)
    shard_tensor = qv.ShardTensor(0)
    shard_tensor.add(device_0_tensor, 0)
    shard_tensor.add(device_1_tensor, 100)
    print(shard_tensor.shape())
    print(shard_tensor.numel())

test_shard_tensor_intra_process()
    
    