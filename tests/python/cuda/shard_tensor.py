import torch
import torch_quiver as qv

def test_shard_tensor_intra_process():
    device_0_tensor = torch.randint(0, 100, (100, 30), device = "cuda:0")
    device_1_tensor = torch.randint(0, 100, (50, 30), device = "cuda:1")
    shard_tensor = qv.ShardTensor([device_0_tensor, device_1_tensor], [100, 50], 0)
    print(shard_tensor.shape())
    print(shard_tensor.numel())

test_shard_tensor_intra_process()
    
    