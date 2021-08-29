import torch
import torch_quiver as qv

def test_shard_tensor_intra_process():
    device_0_tensor = torch.randint(0, 100, (100, 30), device = "cuda:0", dtype = torch.float32)
    device_1_tensor = torch.randint(0, 100, (50, 30), device = "cuda:1", dtype=torch.float32)
    print(f"device_0_tensor device {device_0_tensor.device}\ndevice_1_tensor device {device_1_tensor.device}")
    shard_tensor = qv.ShardTensor(0)
    shard_tensor.add(device_0_tensor, 0)
    shard_tensor.add(device_1_tensor, 100)
    print("shard_tensor shape = ", shard_tensor.shape())
    indices = torch.arange(0, 100).type(torch.long)
    feature = shard_tensor[indices]
    print(f"gathered data shape = {feature.shape}")
    

test_shard_tensor_intra_process()
    
    