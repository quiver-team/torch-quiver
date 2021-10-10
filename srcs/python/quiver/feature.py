import torch
import torch_quiver as torch_qv
import random
import numpy as np
import time
from typing import List
from quiver.shard_tensor import ShardTensor, ShardTensorConfig, Topo
from quiver.utils import reindex_feature

__all__ = ["Feature"]

class Feature:
    def __init__(self, rank, device_list, device_cache_size=0, cache_policy='device_replicate', reorder=None):
        self.device_cache_size = device_cache_size
        self.cache_policy = cache_policy
        self.device_list = device_list
        self.device_tensor_list = {}
        self.numa_tensor_list = {}
        self.rank = rank            
        self.topo = Topo(self.device_list)
        self.reorder = reorder
        self.new_order = None

        self.ipc_handle_ = None
    
    def cal_memory_budget_bytes(self, memory_budget):
        if isinstance(memory_budget, int):
            return memory_budget
        elif isinstance(memory_budget, float):
            memory_budget = int(memory_budget)
        elif isinstance(memory_budget, str):
            if memory_budget.upper().endswith("M") or memory_budget.upper().endswith("MB"):
                end = -1 if memory_budget.upper().endswith("M") else -2
                memory_budget = int(float(memory_budget[:end]) * 1024 * 1024)
            elif memory_budget.upper().endswith("G") or memory_budget.upper().endswith("GB"):
                end = -1 if memory_budget.upper().endswith("G") else -2
                memory_budget = int(float(memory_budget[:end]) * 1024 * 1024 * 1024)
        else:
            raise Exception("memory budget input is not valid")
        return memory_budget


    def cal_size(self, cpu_tensor, cache_memory_budget):
        element_size = cpu_tensor.shape[1] * 4
        cache_size = cache_memory_budget // element_size
        return cache_size

    def partition(self, cpu_tensor, cache_memory_budget):
        
        cache_size = self.cal_size(cpu_tensor, cache_memory_budget)
        return [cpu_tensor[:cache_size], cpu_tensor[cache_size: ]]

    def from_cpu_tensor(self, cpu_tensor):
        if self.cache_policy == "device_replicate":
            cache_memory_budget = self.cal_memory_budget_bytes(self.device_cache_size)
            shuffle_ratio = 0.0
        else:
            cache_memory_budget = self.cal_memory_budget_bytes(self.device_cache_size) * len(self.topo.Numa2Device[0])
            shuffle_ratio = self.cal_size(cpu_tensor, cache_memory_budget) / cpu_tensor.size(0)
        
        print(f"LOG>>> {min(100, int(100 * cache_memory_budget / cpu_tensor.numel() / 4))}% data cached")
        if self.reorder is not None:
            cpu_tensor, new_order = reindex_feature(self.reorder, cpu_tensor, shuffle_ratio)
            self.new_order = new_order.to(self.rank)
        cache_part, self.cpu_part = self.partition(cpu_tensor, cache_memory_budget)
        self.cpu_part = self.cpu_part.clone()
        if cache_part.shape[0] > 0 and self.cache_policy == "device_replicate":
            for device in self.device_list:
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                shard_tensor.append(cache_part, device)
                self.device_tensor_list[device] = shard_tensor

        elif cache_part.shape[0] > 0:
            numa0_device_list = self.topo.Numa2Device[0]
            numa1_device_list = self.topo.Numa2Device[1]

            block_size = self.cal_size(cpu_tensor, cache_memory_budget // len(self.topo.Numa2Device[0]))

            if len(numa0_device_list) > 0:
                print(f"LOG>>> GPU {numa0_device_list} belong to the same NUMA Domain")
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                cur_pos = 0
                for device in numa0_device_list:
                    shard_tensor.append(cache_part[cur_pos: cur_pos + block_size], device)
                    cur_pos += block_size
                    if cur_pos >= block_size * len(self.topo.Numa2Device[0]) or cur_pos >= cache_part.shape[0]:
                        break

                self.numa_tensor_list[0] = shard_tensor
            
            if len(numa1_device_list) > 0:
                print(f"LOG>>> GPU {numa1_device_list} belong to the same NUMA Domain")
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                cur_pos = 0
                for device in numa1_device_list:
                    shard_tensor.append(cache_part[cur_pos: cur_pos + block_size], device)
                    cur_pos += block_size
                    if cur_pos >= block_size * len(self.topo.Numa2Device[0]) or cur_pos >= cache_part.shape[0]:
                        break

                self.numa_tensor_list[1] = shard_tensor

        # 构建CPU Tensor
        if self.cpu_part.numel() > 0:
            if self.cache_policy == "device_replicate":
                shard_tensor = self.device_tensor_list.get(self.rank, None) or ShardTensor(self.rank, ShardTensorConfig({}))
                shard_tensor.append(self.cpu_part, -1)
                self.device_tensor_list[self.rank] = shard_tensor
            else:
                numa_id = self.topo.get_numa_node(self.rank)
                shard_tensor = self.numa_tensor_list.get(numa_id, None) or ShardTensor(self.rank, ShardTensorConfig({}))
                shard_tensor.append(self.cpu_part, -1)
                self.numa_tensor_list[numa_id] = shard_tensor
            
        
    def __getitem__(self, node_idx):
        self.lazy_init_from_ipc_handle()
        node_idx = node_idx.to(self.rank)
        if self.new_order is not None:
            node_idx = self.new_order[node_idx]
        if self.cache_policy == "device_replicate":
            shard_tensor = self.device_tensor_list[self.rank]
            return shard_tensor[node_idx]
        else:
            numa_id = self.topo.get_numa_node(self.rank)
            shard_tensor = self.numa_tensor_list[numa_id]
            return shard_tensor[node_idx]
    
    def size(self, dim):
        self.lazy_init_from_ipc_handle()
        if self.cache_policy == "device_replicate":
            shard_tensor = self.device_tensor_list[self.rank]
            return shard_tensor.size(dim)
        else:
            numa_id = self.topo.get_numa_node(self.rank)
            shard_tensor = self.numa_tensor_list[numa_id]
            return shard_tensor.size(dim)

    @property
    def shape(self):
        self.lazy_init_from_ipc_handle()
        if self.cache_policy == "device_replicate":
            shard_tensor = self.device_tensor_list[self.rank]
            return shard_tensor.shape
        else:
            numa_id = self.topo.get_numa_node(self.rank)
            shard_tensor = self.numa_tensor_list[numa_id]
            return shard_tensor.shape
    

    @property
    def ipc_handle(self):
        return self.ipc_handle_
            
    @ipc_handle.setter
    def ipc_handle(self, ipc_handle):
        self.ipc_handle_ = ipc_handle

    def share_ipc(self):
        gpu_ipc_handle_dict = {}
        if self.cache_policy == "device_replicate":
            for device in self.device_tensor_list:
                gpu_ipc_handle_dict[device] = self.device_tensor_list[device].share_ipc()[0]
        else:
            for numa_node in self.numa_tensor_list:
                gpu_ipc_handle_dict[numa_node] = self.numa_tensor_list[numa_node].share_ipc()[0]
        #self.cpu_part = torch.zeros([3, 600])
        return gpu_ipc_handle_dict, self.cpu_part, self.device_list, self.device_cache_size, self.cache_policy
    
    def from_gpu_ipc_handle_dict(self, gpu_ipc_handle_dict, cpu_tensor):
        if self.cache_policy == "device_replicate":
            ipc_handle = gpu_ipc_handle_dict.get(self.rank, []), cpu_tensor, ShardTensorConfig({})
            shard_tensor = ShardTensor.new_from_share_ipc(ipc_handle, self.rank)
            self.device_tensor_list[self.rank] = shard_tensor
            
        else:
            numa_node = self.topo.get_numa_node(self.rank)
            ipc_handle = gpu_ipc_handle_dict.get(numa_node, []), cpu_tensor, ShardTensorConfig({})
            shard_tensor = ShardTensor.new_from_share_ipc(ipc_handle, self.rank)
            self.numa_tensor_list[numa_node] = shard_tensor
        
        self.cpu_part = cpu_tensor
        
        
    @classmethod
    def new_from_ipc_handle(cls, rank, ipc_handle):
        gpu_ipc_handle_dict, cpu_part, device_list, device_cache_size, cache_policy = ipc_handle
        feature = cls(rank, device_list, device_cache_size, cache_policy)
        feature.from_gpu_ipc_handle_dict(gpu_ipc_handle_dict, cpu_part)
        return feature
    
    @classmethod
    def lazy_from_ipc_handle(cls, ipc_handle):
        gpu_ipc_handle_dict, cpu_part, device_list, device_cache_size, cache_policy = ipc_handle
        feature = cls(device_list[0], device_list, device_cache_size, cache_policy)
        feature.ipc_handle = ipc_handle
        return feature
    
    def lazy_init_from_ipc_handle(self):
        if self.ipc_handle is None:
            return 
        
        self.rank = torch.cuda.current_device()
        gpu_ipc_handle_dict, cpu_part, device_list, device_cache_size, cache_policy = self.ipc_handle
        self.from_gpu_ipc_handle_dict(gpu_ipc_handle_dict, cpu_part)
        self.ipc_handle = None
