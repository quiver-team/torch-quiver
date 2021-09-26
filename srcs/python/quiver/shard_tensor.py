import torch_quiver as torch_qv
import torch
import random
from typing import List
import time

class Offset:
    def __init__(self, start, end):
        self.start_ = start
        self.end_ = end
    
    @property
    def start(self):
        return self.start_
    
    @property
    def end(self):
        return self.end_

class DeviceCollectionJob:
    def __init__(self, part_orders, request_nodes):
        self.part_orders_ = part_orders
        self.request_nodes_ = request_nodes
    
    @property
    def part_orders(self):
        return self.part_orders_
    
    @property
    def request_nodes(self):
        return self.request_nodes_
    


class ShardTensorConfig:
    
    def __init__(self, device_memory_budget, tensor_offset_device=None):
        self.offset_array_ = []
        if tensor_offset_device is None:
            self.tensor_offset_device = {}
        else:
            self.tensor_offset_device = tensor_offset_device

        self.device_memory_budget = device_memory_budget
        self.device_list_ = None
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
    
    @property
    def device_list(self):
        if self.device_list_ is None:
            self.device_list_ = list(self.device_memory_budget.keys())
        return self.device_list_
    
    @device_list.setter
    def device_list(self, device_list):
        self.device_list_ = device_list
    
    @property
    def offset_array(self):
        return self.offset_array_
    
    @offset_array.setter
    def offset_array(self, tmp_array):
        self.offset_array_ = torch.as_tensor(tmp_array, dtype=torch.int32)
    

class ShardTensor:
    def __init__(self, current_device: int, shard_tensor_config: ShardTensorConfig):
        self.shard_tensor = torch_qv.ShardTensor(current_device)
        self.current_device = current_device
        self.shard_tensor_config = shard_tensor_config
        self.device_stream = {}

        # cpu part
        self.cpu_tensor = None

        # current stream
        self.current_stream = torch.cuda.Stream(self.current_device)



    
    def partition(self, tensor, memory_budget):
        """
        Args:
            tensor: pytorch cpu tensor
            memory_budget: memory size in bytes
            
        """
        # FIXME we assume it is Float tensor
        element_size = tensor.stride(0) * 4
        return memory_budget // element_size
    
    
    def from_cpu_tensor(self, tensor):
        cur_pos = 0
        size = 0
        offset_array = []
        # allocate for GPU
        for  device_id, memory_budget in self.shard_tensor_config.device_memory_budget.items():
            size = self.partition(tensor, memory_budget)
            size  = min(size, tensor.shape[0] - cur_pos)
            self.shard_tensor.append(tensor[cur_pos: cur_pos + size], device_id)
            device_offset = Offset(cur_pos, cur_pos + size)
            self.shard_tensor_config.tensor_offset_device[device_id] = device_offset
            cur_pos += size
            offset_array.append(cur_pos)
            print(f"LOG >>> Assign {int(100 * size * 1.0 / tensor.shape[0])}% data to {device_id}")
            if cur_pos > tensor.shape[0]:
                break
        if cur_pos < tensor.shape[0]:
            # allocate for CPU
            self.cpu_tensor = tensor[cur_pos:].clone()
            self.cpu_tensor.share_memory_()
            self.shard_tensor.append(self.cpu_tensor, -1)
            print(f"LOG >>> Assign {100 - int(100 * cur_pos * 1.0 / tensor.shape[0])}% data to CPU")
            
        # init config 
        self.shard_tensor_config.offset_array = offset_array
        self.offset_array = self.shard_tensor_config.offset_array.to(self.current_device)
    
    def collect_device(self, part_orders, request_nodes, inter_device, wait_streams, wait_results):

        if len(wait_results) > 0:
            with torch.cuda.stream(self.current_stream):
                wait_streams[-1].synchronize()
                wait_results[-1][1] = wait_results[-1][1].to(self.current_device)

       
        with torch.cuda.device(inter_device):
            if self.device_stream.get(inter_device, None) is None:
                self.device_stream[inter_device] = torch.cuda.Stream(inter_device)
            with torch.cuda.stream(self.device_stream[inter_device]):
                request_nodes = request_nodes.to(inter_device)
                result = self.shard_tensor[request_nodes]
        wait_streams.append(self.device_stream[inter_device])
        wait_results.append([part_orders, result])
    
    def __getitem__(self, nodes):
        
        #start_time = time.time()
        if self.device_stream.get(self.current_device, None) is None:
            self.device_stream[self.current_device] = torch.cuda.Stream(self.current_device)
        
        if len(self.shard_tensor_config.device_list)> 0:
            with torch.cuda.stream(self.current_stream):
                sorted_nodes, sorted_order = torch.sort(nodes)
                offsets = torch.searchsorted(sorted_nodes, self.offset_array)
        #print(f"after sort launch {time.time() - start_time}")


        with torch.cuda.stream(self.device_stream[self.current_device]):
            feature = self.shard_tensor[nodes]
        
        #print(f"after local collect launch {time.time() - start_time}")
        

        dispatch_book = {}
        if len(self.shard_tensor_config.device_list)> 0 :
            self.current_stream.synchronize()
            device_list = self.shard_tensor_config.device_memory_budget.keys()
            start = 0
            end = 0
            index = 0
            for device, offset in zip(device_list, offsets):
                if device == self.current_device:
                    start = offset
                    index += 1
                    continue
                end = offset
                dispatch_book[device] = DeviceCollectionJob(sorted_order[start:end], sorted_nodes[start:end])
                start = end
                index += 1
        #print(f"after preprocess time = {time.time() - start_time}")
        
        wait_streams = []
        wait_results = []
        device_list = list(dispatch_book.keys())
        
        if len(device_list) > 0:
            device = device_list.pop()
            self.collect_device(dispatch_book[device].part_orders, dispatch_book[device].request_nodes, device, wait_streams, wait_results)
            #print(f"after device {device} collection kernel launch {time.time() - start_time}")
        
        if len(device_list) > 0:
            device = device_list.pop()
            self.collect_device(dispatch_book[device].part_orders, dispatch_book[device].request_nodes, device, wait_streams, wait_results)
            #print(f"after device {device} collection kernel launch {time.time() - start_time}")
        
        if len(device_list) > 0:
            device = device_list.pop()
            self.collect_device(dispatch_book[device].part_orders, dispatch_book[device].request_nodes, device, wait_streams, wait_results)
            #print(f"after device {device} collection kernel launch {time.time() - start_time}")
        
        if len(wait_streams) > 0:
            self.current_stream.synchronize()
            wait_streams[-1].synchronize()
            wait_results[-1][1] = wait_results[-1][1].to(self.current_device)
            for result  in wait_results:
                feature[result[0]] = result[1]
            
        self.device_stream[self.current_device].synchronize()
        #print(f"after all synchronize {time.time() - start_time}")
        

        return feature
    
    
    @property
    def shape(self):
        return self.shard_tensor.shape()
    
    @property
    def device(self):
        return self.current_device
    
    def share_ipc(self):
        items = self.shard_tensor.share_ipc()
        gpu_part_ipc_list = [item.share_ipc() for item in items]

        return gpu_part_ipc_list, self.cpu_tensor, self.shard_tensor_config

    def from_ipc_handle(self, gpu_ipc_list, cpu_tensor):
        for gpu_ipc in gpu_ipc_list:
            gpu_item = torch_qv.ShardTensorItem()
            gpu_item.from_ipc(gpu_ipc)
            self.shard_tensor.append(gpu_item)
        if cpu_tensor is not None:
            self.cpu_tensor = cpu_tensor
            self.shard_tensor.append(cpu_tensor, -1)
        self.offset_array = self.shard_tensor_config.offset_array.to(self.current_device)

    @classmethod
    def new_from_share_ipc(cls, ipc_handles):
         gpu_part_ipc_list, cpu_tensor, shard_tensor_config = ipc_handles
         current_device = torch.cuda.current_device()
         shard_tensor = cls(current_device, shard_tensor_config)
         shard_tensor.from_ipc_handle(gpu_part_ipc_list, cpu_tensor)
         return shard_tensor



    
    
