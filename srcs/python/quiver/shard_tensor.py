import torch_quiver as torch_qv
import torch
import random
from typing import List
import time


def color_mat(access_book, device_list):
    device_count = access_book.shape[0]

    device2numa = dict.fromkeys(device_list, -1)
    numa2device = {0: [], 1: []}
    current_numa = 0
    for src_device_idx in range(device_count):
        src_device = device_list[src_device_idx]
        if(device2numa[src_device] == -1):
            device2numa[src_device] = current_numa
            numa2device[current_numa].append(src_device)
            current_numa += 1
            for dst_device_idx in range(device_count):
                if(dst_device_idx != src_device_idx and access_book[src_device_idx, dst_device_idx] == 1):
                    dst_device = device_list[dst_device_idx]
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
                if torch_qv.can_device_access_peer(src_device, dst_device):
                    access_book[src_index][dst_index] = 1
                    access_book[dst_index][src_index] = 1
        self.Device2Numa, self.Numa2Device = color_mat(access_book, device_list)
        
    
    def get_numa_node(self, device_id: int):
        return self.Device2Numa[device_id]
    
    def random_pick_device_from_numa(self, numa_id):
        return random.choice(self.Numa2Device[numa_id])

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
    
    def __init__(self, device_memory_budget, tensor_offset_numa=None, tensor_offset_device=None):
        self.offset_array_ = []
        if tensor_offset_numa is None:
            self.tensor_offset_device = {}
            self.tensor_offset_numa = {}
        else:
            self.tensor_offset_device = tensor_offset_device
            self.tensor_offset_numa = tensor_offset_numa

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
        device_list = shard_tensor_config.device_list
        device_list = device_list if current_device in device_list else device_list + [current_device]
        self.topo = Topo(device_list)
        self.shard_tensor_config.device_list = self.topo.Numa2Device[0] + self.topo.Numa2Device[1]
        
        # we assume there are at most 2 Numa Nodes
        self.current_numa = self.topo.get_numa_node(current_device)
        self.device_stream = {}

        # cpu part
        self.cpu_tensor = None

    
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
        cur_pos = 0
        size = 0
        numa_size = [0, 0]
        offset_array = []
        # 首先给GPU分配数据
        for  device_id, memory_budget in self.shard_tensor_config.device_memory_budget.items():
            size = self.partition(tensor, memory_budget)
            size  = min(size, tensor.shape[0] - cur_pos)
            self.shard_tensor.append(tensor[cur_pos: cur_pos + size], device_id)
            device_offset = Offset(cur_pos, cur_pos + size)
            self.shard_tensor_config.tensor_offset_device[device_id] = device_offset
            cur_pos += size
            offset_array.append(cur_pos)
            numa_node = self.topo.get_numa_node(device_id)
            numa_size[numa_node] += size
            print(f"LOG >>> Assign {int(100 * size * 1.0 / tensor.shape[0])}% data to {device_id}")
            if cur_pos > tensor.shape[0]:
                break
        if cur_pos < tensor.shape[0]:
            # 接着继续给CPU分配数据
            self.cpu_tensor = tensor[cur_pos:].clone()
            self.cpu_tensor.share_memory_()
            self.shard_tensor.append(self.cpu_tensor, -1)
            print(f"LOG >>> Assign {100 - int(100 * cur_pos * 1.0 / tensor.shape[0])}% data to CPU")
            
        # init config 
        self.shard_tensor_config.tensor_offset_numa[0] = Offset(0, numa_size[0])
        self.shard_tensor_config.tensor_offset_numa[1] = Offset(numa_size[0], numa_size[0] + numa_size[1])
        self.shard_tensor_config.offset_array = offset_array
        self.offset_array = self.shard_tensor_config.offset_array.to(self.current_device)

    def collect_device(self, input_orders, nodes, inter_device, wait_streams, wait_results):
        request_nodes_mask = (nodes >= self.shard_tensor_config.tensor_offset_device[inter_device].start) & (nodes < self.shard_tensor_config.tensor_offset_device[inter_device].end)
        request_nodes = torch.masked_select(nodes, request_nodes_mask)
        part_orders = torch.masked_select(input_orders, request_nodes_mask)
        with torch.cuda.device(inter_device):
            if self.device_stream.get(inter_device, None) is None:
                self.device_stream[inter_device] = torch.cuda.Stream(inter_device)
            with torch.cuda.stream(self.device_stream[inter_device]):
                request_nodes = request_nodes.to(inter_device, non_blocking=True)
                result = self.shard_tensor[request_nodes]
                result = result.to(self.current_device, non_blocking=True)
        wait_streams.append(self.device_stream[inter_device])
        wait_results.append((part_orders, result))
    
    def collect_devicev2(self, part_orders, request_nodes, inter_device, wait_streams, wait_results):

        
        with torch.cuda.device(inter_device):
            if self.device_stream.get(inter_device, None) is None:
                self.device_stream[inter_device] = torch.cuda.Stream(inter_device)
            with torch.cuda.stream(self.device_stream[inter_device]):
                request_nodes = request_nodes.to(inter_device, non_blocking=True)
                result = self.shard_tensor[request_nodes]
                result = result.to(self.current_device, non_blocking=True)
        wait_streams.append(self.device_stream[inter_device])
        wait_results.append((part_orders, result))

    def __getitem__(self, nodes):


        if self.device_stream.get(self.current_device, None) is None:
            self.device_stream[self.current_device] = torch.cuda.Stream(self.current_device)

        with torch.cuda.stream(self.device_stream[self.current_device]):
            feature = self.shard_tensor[nodes]
        
        input_orders = torch.arange(nodes.size(0), dtype=torch.long, device = self.current_device)

        # call inter request, we unfold for loop 
        inter_numa_devices = self.topo.Numa2Device[1 - self.current_numa]
        intra_numa_devices = self.topo.Numa2Device[self.current_numa]
        wait_streams = []
        wait_results = []
        for device in inter_numa_devices:
            if self.shard_tensor_config.tensor_offset_device.get(device, None) is not None:
                self.collect_device(input_orders, nodes, device, wait_streams, wait_results)
        
        for device in intra_numa_devices:
            if device == self.current_device:
                continue
            if self.shard_tensor_config.tensor_offset_device.get(device, None) is not None:
                self.collect_device(input_orders, nodes, device, wait_streams, wait_results)

        for stream, result  in zip(wait_streams, wait_results):
            stream.synchronize()
            feature[result[0]] = result[1]
        self.device_stream[self.current_device].synchronize()

        return feature
    
    def collect(self, nodes):
        dispatch_book = {}
        start_time = time.time()
        sorted_nodes, sorted_order = torch.sort(nodes)
        offsets = torch.searchsorted(sorted_nodes, self.offset_array, right=True)
        device_list = self.shard_tensor_config.device_memory_budget.keys()
        start = 0
        end = 0
        for device, offset in zip(device_list, offsets):
            if device == self.current_device:
                start = offset
                continue
            dispatch_book[device] = DeviceCollectionJob(sorted_order[start:end], sorted_nodes[start:end])
            start = offset
        print(f"preprocess time = {time.time() - start_time}")

        if self.device_stream.get(self.current_device, None) is None:
            self.device_stream[self.current_device] = torch.cuda.Stream(self.current_device)

        with torch.cuda.stream(self.device_stream[self.current_device]):
            feature = self.shard_tensor[nodes]
        
        wait_streams = []
        wait_results = []
        for device in dispatch_book:
            if device == self.current_device:
                continue
            self.collect_devicev2(dispatch_book[device].part_orders, dispatch_book[device].request_nodes, device, wait_streams, wait_results)
        
        for stream, result  in zip(wait_streams, wait_results):
            stream.synchronize()
            feature[result[0]] = result[1]
        
        self.device_stream[self.current_device].synchronize()
        

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

    @classmethod
    def new_from_share_ipc(cls, ipc_handles):
         gpu_part_ipc_list, cpu_tensor, shard_tensor_config = ipc_handles
         current_device = torch.cuda.current_device()
         shard_tensor = cls(current_device, shard_tensor_config)
         shard_tensor.from_ipc_handle(gpu_part_ipc_list, cpu_tensor)
         return shard_tensor



    
    
