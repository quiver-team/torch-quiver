import torch_quiver as torch_qv
import torch
import random
from typing import List


def color_mat(access_book):
    device_count = access_book.shape[0]
    device2numa = [-1] * device_count
    numa2device = {0: [], 1: []}
    current_numa = 0
    for src_device in range(device_count):
        if(device2numa[src_device] == -1):
            device2numa[src_device] = current_numa
            numa2device[device2numa[src_device]].append(src_device)
            current_numa += 1
            for dst_device in range(device_count):
                if(dst_device != src_device and access_book[src_device, dst_device] == 1):
                    device2numa[dst_device] = device2numa[src_device]
                    numa2device[device2numa[src_device]].append(dst_device)
    
    return device2numa, numa2device
            
    
class Topo:
    
    Numa2Device = {}
    Device2Numa = {}
    
    def __init__(self, device_list: List[int]) -> None:
        access_book = torch.zeros((len(device_list), len(device_list)))
        for src_index, src_device in enumerate(device_list):
            for dst_index, dst_device in enumerate(device_list):
                if torch._C._cuda_canDeviceAccessPeer(src_device, dst_device) and torch._C._cuda_canDeviceAccessPeer(dst_device, src_device):
                    access_book[src_index][dst_index] = 1
                    access_book[dst_index][src_index] = 1
        self.Device2Numa, self.Numa2Device = color_mat(access_book)
        
    
    def get_numa_node(self, device_id: int):
        return self.Device2Numa[device_id]
    
    def random_pick_device_from_numa(self, numa_id):
        return random.choice(self.Numa2Device[numa_id])

class ShardTensorConfig:
    device_memory_budget = {}
    tensor_offset_device = []
    tensor_offset_numa = []
    
    def __init__(self, device_memory_budget):
        self.device_memory_budget = device_memory_budget
        for device in device_memory_budget:
            if isinstance(self.device_memory_budget[device], int):
                continue
            elif isinstance(self.device_memory_budget[device], float):
                self.device_memory_budget[device] = int(self.device_memory_budget[device])
            
            elif isinstance(self.device_memory_budget[device], str):
                if self.device_memory_budget[device].upper().endswith("M") or self.device_memory_budget[device].upper().endswith("MB"):
                    end = -1 if self.device_memory_budget[device].upper().endswith("M") else -2
                    self.device_memory_budget[device] = int(float(self.device_memory_budget[device][:end]) * 1024 * 1024)
                
                elif self.device_memory_budget[device].upper().endswith("G") or self.device_memory_budget[device].upper().endswith("GB"):
                    end = -1 if self.device_memory_budget[device].upper().endswith("G") else -2
                    self.device_memory_budget[device] = int(float(self.device_memory_budget[device][:end]) * 1024 * 1024 * 1024)
            else:
                raise Exception("memory budget input is not valid")
            print(f"LOG >>> Memory Budge On {device} is {self.device_memory_budget[device] // 1024 // 1024}MB")
    
    def reorder_device(self, device_order):
        tmp_device_memory_budget = {}
        for device in device_order:
            tmp_device_memory_budget[device] = self.device_memory_budget[device]
        self.device_memory_budget = tmp_device_memory_budget
    
    @property
    def device_list(self):
        return list(self.device_memory_budget.keys())
    

class ShardTensor:
    def __init__(self, current_device: int, shard_tensor_config: ShardTensorConfig):
        self.shard_tensor = torch_qv.ShardTensor(current_device)
        self.current_device = current_device
        self.shard_tensor_config = shard_tensor_config
        self.topo = Topo(shard_tensor_config.device_list)
        self.shard_tensor_config.reorder_device(self.topo.Numa2Device[0] + self.topo.Numa2Device[1])
        
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
        for  device_id, memory_budget in self.shard_tensor_config.device_memory_budget.items():
            size = self.partition(tensor, memory_budget)
            self.shard_tensor.append(tensor[offset: offset + size], device_id)
            offset += size
            numa_node = self.topo.get_numa_node(device_id)
            numa_size[numa_node] += size
            print(f"LOG >>> Assign {int(100 * size * 1.0 / tensor.shape[0])}% data to {device_id}")
            if offset > tensor.shape[0]:
                break
        if offset < tensor.shape[0]:
            # 接着继续给CPU分配数据
            self.shard_tensor.append(tensor[offset:], -1)
            print(f"LOG >>> Assign {100 - int(100 * offset * 1.0 / tensor.shape[0])}% data to CPU")
            
        # init config 
        self.shard_tensor_config.tensor_offset_numa.append(numa_size[0])
        self.shard_tensor_config.tensor_offset_numa.append(numa_size[0] + numa_size[1])
    
        
    def __getitem__(self, nodes):

        input_orders = torch.arange(nodes.size(0), dtype=torch.long, device=nodes.device)
        # async
        feature = self.shard_tensor[nodes]
        # call request
        
        if self.current_numa == 0:
            request_nodes_mask = (nodes >= self.shard_tensor_config.tensor_offset_numa[0]) & (nodes < self.shard_tensor_config.tensor_offset_numa[1])
        else:
            request_nodes_mask = nodes < self.shard_tensor_config.tensor_offset_numa[0]
        request_nodes = torch.masked_select(nodes, request_nodes_mask)
        part_orders = torch.masked_select(input_orders, request_nodes_mask)
        # ptr0, ptr1, ptr2, ptr3, ptr_cpu
        if request_nodes.shape[0] > 0 :
            # chosen_device = 2
            chosen_device = self.topo.random_pick_device_from_numa(1 - self.current_numa)
            # access ptr2, ptr3 on device 2 to collect data
            with torch.device(chosen_device):
                request_nodes = request_nodes.to(chosen_device)
                result = self.shard_tensor[request_nodes]
            result.to(self.current_device)
            feature[part_orders] = result
        
        return feature
    
    @property
    def shape(self):
        return self.shard_tensor.shape()
    
    @property
    def device(self):
        return self.current_device
