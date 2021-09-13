import torch_quiver as torch_qv
import torch
from typing import List

class Topo:
    
    Numa2Device = {}
    Device2Numa = {}
    
    def __init__(self, device_list: List[int]) -> None:
        pass
    
    def get_numa_node(self, device_id):
        return 1
    
    def random_pick_device_from_numa(self, numa_id):
        return 1

class ShardTensorConfig:
    device_memory_budget = {}
    tensor_offset_device = []
    tensor_offset_numa = []
    

class ShardTensor:
    def __init__(self, current_device: int, shard_tensor_config: ShardTensorConfig):
        self.shard_tensor = torch_qv.ShardTensor(current_device)
        self.current_device = current_device
        self.shard_tensor_config = shard_tensor_config
        self.topo = Topo()
        # we assume there are at most 2 Numa Node
        self.current_numa = self.topo.get_numa_node(current_device)
        
    
    def partition(self, tensor, memory_budget):
        """
        Args:
            tensor: pytorch cpu tensor
            memory_budget: memory size in bytes
            
        """
        # 暂时先假设为float tensor
        element_size = tensor.stride(0) * 4
        return memory_budget // element_size
    
    
    def from_cpu_tensor(self, tensor):
        # 我们假设device按照NUMA顺序已经排序
        offset = 0
        size = 0
        numa_size = [0, 0]
        # 首先给GPU分配数据
        for index, device_id, memory_budget in enumerate(self.shard_tensor_config.device_memory_budget.items()):
            size = self.partition(tensor, memory_budget)
            self.shard_tensor.append(tensor[offset: offset + size], device_id)
            offset += size
            numa_node = self.topo.get_numa_node(device_id)
            numa_size[numa_node] += size
        # 接着继续给CPU分配数据
        self.shard_tensor.append(tensor[offset:], -1)
        
        # init config 
        self.shard_tensor_config.tensor_offset_numa[0] = numa_size[0]
        self.shard_tensor_config.tensor_offset_numa[0] = numa_size[0] + numa_size[1]
    
        
    def __getitem__(self, nodes):
        input_orders = torch.arange(nodes.size(0), dtype=torch.long, device=nodes.device)
        # async
        feature = self.shard_tensor[nodes]
        # call request
        chosen_device = self.topo.random_pick_device_from_numa(1 - self.current_numa)
        if self.current_numa == 0:
            request_nodes_mask = nodes[nodes >= self.shard_tensor_config.tensor_offset_numa[0]]
        else:
            request_nodes_mask = nodes[nodes < self.shard_tensor_config.tensor_offset_numa[0]]
        
        if request_nodes_mask.shape[0] > 0 :
            request_nodes = torch.masked_select(nodes, request_nodes_mask)
            part_orders = torch.masked_select(input_orders, request_nodes_mask)
            
            with torch.device(chosen_device):
                result = self.shard_tensor[request_nodes]
            result.to(self.current_device)
            feature[part_orders] = result
        
        return feature
    
            
        
        