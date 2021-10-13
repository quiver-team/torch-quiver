import torch
import torch_quiver as torch_qv
import random
import numpy as np
import time
from typing import List
from quiver.shard_tensor import ShardTensor, ShardTensorConfig, Topo
from quiver.utils import reindex_feature
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import os 
import sys
import quiver
import torch.distributed as dist


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
    def __init__(self, rank, device_list, device_cache_size=0, cache_policy='device_replicate', csr_topo=None):
        self.device_cache_size = device_cache_size
        self.cache_policy = cache_policy
        self.device_list = device_list
        self.device_tensor_list = {}
        self.numa_tensor_list = {}
        self.rank = rank            
        self.topo = Topo(self.device_list)
        self.csr_topo = csr_topo
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
        if self.csr_topo is not None:
            print("Create")
            cpu_tensor, self.csr_topo.feature_order = reindex_feature(self.csr_topo, cpu_tensor, shuffle_ratio)
            self.feature_order = self.csr_topo.feature_order.to(self.rank)
            print("Done Create")
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
                for idx, device in enumerate(numa0_device_list):
                    if idx == len(numa0_device_list) - 1:
                        shard_tensor.append(cache_part[cur_pos:], device)
                    else:

                        shard_tensor.append(cache_part[cur_pos: cur_pos + block_size], device)
                        cur_pos += block_size
                    
                self.numa_tensor_list[0] = shard_tensor
            
            if len(numa1_device_list) > 0:
                print(f"LOG>>> GPU {numa1_device_list} belong to the same NUMA Domain")
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                cur_pos = 0
                for idx, device in enumerate(numa1_device_list):
                    if idx == len(numa1_device_list) - 1:
                        shard_tensor.append(cache_part[cur_pos:], device)
                    else:

                        shard_tensor.append(cache_part[cur_pos: cur_pos + block_size], device)
                        cur_pos += block_size

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
        if self.feature_order is not None:
            node_idx = self.feature_order[node_idx]
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
        
        return gpu_ipc_handle_dict, self.cpu_part, self.device_list, self.device_cache_size, self.cache_policy, self.csr_topo
    
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
        gpu_ipc_handle_dict, cpu_part, device_list, device_cache_size, cache_policy, csr_topo = ipc_handle
        feature = cls(rank, device_list, device_cache_size, cache_policy)
        feature.from_gpu_ipc_handle_dict(gpu_ipc_handle_dict, cpu_part)
        if csr_topo is not None:
            feature.feature_order = csr_topo.feature_order.to(rank)
        self.csr_topo = csr_topo
        return feature
    
    @classmethod
    def lazy_from_ipc_handle(cls, ipc_handle):
        gpu_ipc_handle_dict, cpu_part, device_list, device_cache_size, cache_policy, _ = ipc_handle
        feature = cls(device_list[0], device_list, device_cache_size, cache_policy)
        feature.ipc_handle = ipc_handle
        return feature
    
    def lazy_init_from_ipc_handle(self):
        if self.ipc_handle is None:
            return 
        
        self.rank = torch.cuda.current_device()
        gpu_ipc_handle_dict, cpu_part, device_list, device_cache_size, cache_policy, csr_topo = self.ipc_handle
        self.from_gpu_ipc_handle_dict(gpu_ipc_handle_dict, cpu_part)
        self.csr_topo = csr_topo
        if csr_topo is not None:
            self.feature_order = csr_topo.feature_order.to(self.rank)

        self.ipc_handle = None

from multiprocessing.reduction import ForkingPickler

def rebuild_feature(ipc_handle):
    print("check rebuild")
    feature = Feature.lazy_from_ipc_handle(ipc_handle)
    return feature

def reduce_feature(feature):
    
    ipc_handle = feature.share_ipc()
    return (rebuild_feature, (ipc_handle, ))


def rebuild_pyg_sampler(cls, ipc_handle):
    sampler = cls.lazy_from_ipc_handle(ipc_handle)
    return sampler
    

def reduce_pyg_sampler(sampler):
    ipc_handle = sampler.share_ipc()
    return (rebuild_pyg_sampler, (type(sampler), ipc_handle, ))
  



def init_reductions():
    ForkingPickler.register(Feature, reduce_feature)

def test_feature_basic():
    rank = 0
    
    NUM_ELEMENT = 1000000
    SAMPLE_SIZE = 80000
    FEATURE_DIM = 600

    #########################
    # Init With Numpy
    ########################
    torch.cuda.set_device(rank)


    host_tensor = np.random.randint(
        0, high=10, size=(2 * NUM_ELEMENT, FEATURE_DIM))
    tensor = torch.from_numpy(host_tensor).type(torch.float32)
    host_indice = np.random.randint(0, 2 * NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)
    print("host data size", host_tensor.size * 4 // 1024  // 1024, "MB")


    device_indices = indices.to(rank)

    ############################
    # define a quiver.Feature
    ###########################
    feature = quiver.Feature(rank=rank, device_list=[0, 1, 2, 3], device_cache_size="0.9G", cache_policy="numa_replicate")
    feature.from_cpu_tensor(tensor)

    ####################
    # Indexing 
    ####################
    res = feature[device_indices]
    
    start = time.time()
    res = feature[device_indices]
    consumed_time = time.time() - start
    res = res.cpu().numpy()
    feature_gt = tensor[indices].numpy()
    print("Correctness Check : ", np.array_equal(res, feature_gt))
    print(
        f"Process {os.getpid()}: TEST SUCCEED!, With Memory Bandwidth = {res.size * 4 / consumed_time / 1024 / 1024 / 1024} GB/s, consumed {consumed_time}s")

def child_proc(rank, world_size, host_tensor, feature):
    torch.cuda.set_device(rank)
    print(f"Process {os.getpid()}: check current device {torch.cuda.current_device()}")
    NUM_ELEMENT = host_tensor.shape[0]
    SAMPLE_SIZE = 80000
    device_tensor = host_tensor.to(rank)
    bandwidth = []
    for _ in range(30):
        device_indices = torch.randint(0, NUM_ELEMENT - 1, (SAMPLE_SIZE, ), device=rank)
        torch.cuda.synchronize()
        start = time.time()
        res = feature[device_indices]
        consumed_time = time.time() - start
        bandwidth.append(res.numel() * 4 / consumed_time / 1024 / 1024 / 1024)
        assert torch.equal(res, device_tensor[device_indices])
    print("Correctness check passed")
    print(
        f"Process {os.getpid()}: TEST SUCCEED!, With Memory Bandwidth = {np.mean(np.array(bandwidth[1:]))} GB/s, consumed {consumed_time}s, res size {res.numel() * 4 / 1024 / 1024 / 1024}GB")

def test_ipc():
    rank = 0
    
    NUM_ELEMENT = 1000000
    FEATURE_DIM = 600

    #########################
    # Init With Numpy
    ########################
    torch.cuda.set_device(rank)

    host_tensor = np.random.randint(
        0, high=10, size=(2 * NUM_ELEMENT, FEATURE_DIM))
    tensor = torch.from_numpy(host_tensor).type(torch.float32)
    print("host data size", host_tensor.size * 4 // 1024  // 1024, "MB")


    ############################
    # define a quiver.Feature
    ###########################

    feature = quiver.Feature(rank=rank, device_list=[0, 1], device_cache_size=0, cache_policy="numa_replicate")
    feature.from_cpu_tensor(tensor)
    world_size = 2
    mp.spawn(
        child_proc,   
        args=(world_size, tensor, feature),
        nprocs=world_size,
        join=True
    )

def child_proc_real_data(rank, feature, host_tensor):
    NUM_ELEMENT = 2000000
    SAMPLE_SIZE = 800000
    bandwidth = []
    torch.cuda.set_device(rank)
    device_tensor = host_tensor.to(rank)
    for _ in range(300):
        device_indices = torch.randint(0, NUM_ELEMENT - 1, (SAMPLE_SIZE, ), device=rank)
        torch.cuda.synchronize()
        start = time.time()
        res = feature[device_indices]
        consumed_time = time.time() - start
        bandwidth.append(res.numel() * 4 / consumed_time / 1024 / 1024 / 1024)
        assert torch.equal(device_tensor[device_indices], res)
    print("Correctness check passed")
    print(
        f"Process {os.getpid()}: TEST SUCCEED!, With Memory Bandwidth = {np.mean(np.array(bandwidth[1:]))} GB/s, consumed {consumed_time}s, res size {res.numel() * 4 / 1024 / 1024 / 1024}GB")

def test_ipc_with_real_data():
    from ogb.nodeproppred import PygNodePropPredDataset
    root = "/home/dalong/data/products"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    data = dataset[0]

    world_size = torch.cuda.device_count()
    
    ##############################
    # Create Sampler And Feature
    ##############################
    csr_topo = quiver.CSRTopo(data.edge_index)
    feature = torch.zeros(data.x.shape)
    feature[:] = data.x
    quiver_feature = Feature(rank=0, device_list=list(range(world_size)), device_cache_size="200M", cache_policy="device_replicate", csr_topo=csr_topo)
    quiver_feature.from_cpu_tensor(feature)

    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(
        child_proc_real_data,
        args=(quiver_feature, feature),
        nprocs=world_size,
        join=True
    )

def normal_test():
    rank = 0
    
    NUM_ELEMENT = 1000000
    FEATURE_DIM = 600
    SAMPLE_SIZE = 80000

    #########################
    # Init With Numpy
    ########################
    torch.cuda.set_device(rank)

    host_tensor = np.random.randint(
        0, high=10, size=(2 * NUM_ELEMENT, FEATURE_DIM))
    tensor = torch.from_numpy(host_tensor).type(torch.float32)

    host_indice = np.random.randint(0, 2 * NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)

    tensor.to(rank)
    torch.cuda.synchronize()

    start = time.time()
    feature = tensor[indices]
    feature = feature.to(rank)
    torch.cuda.synchronize()
    consumed_time = time.time() - start

    print(
        f"Process {os.getpid()}: TEST SUCCEED!, With Memory Bandwidth = {feature.numel() * 4 / consumed_time / 1024 / 1024 / 1024} GB/s, consumed {consumed_time}s")

def test_paper100M():
    dataset = torch.load("/data/papers/ogbn_papers100M/quiver_preprocess/paper100M.pth")
    csr_topo = dataset["csr_topo"]
    feature = dataset["sorted_feature"]
    NUM_ELEMENT = feature.shape[0]
    SAMPLE_SIZE = 80000
    world_size = 4
    rank = 0
    dataset["label"] = torch.from_numpy(dataset["label"])
    dataset["num_features"] = feature.shape[1]
    dataset["num_classes"] = 172
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], 0, mode="UVA")
    quiver_feature = quiver.Feature(rank=0, device_list=list(range(world_size)), device_cache_size="12G", cache_policy="numa_replicate")
    quiver_feature.from_cpu_tensor(feature)

    device_indices = torch.randint(0, NUM_ELEMENT - 1, (SAMPLE_SIZE, ), device=rank)
    res = quiver_feature[device_indices]

    start = time.time()
    res = quiver_feature[device_indices]
    consumed_time = time.time() - start
    print(
        f"Process {os.getpid()}: TEST SUCCEED!, With Memory Bandwidth = {res.numel() * 4 / consumed_time / 1024 / 1024 / 1024} GB/s, consumed {consumed_time}s")





if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch_qv.init_p2p([0,1,2,3])
    test_paper100M()
    #init_reductions()
    #test_feature_basic()
    #test_ipc()
    #normal_test()
    #test_ipc_with_real_data()

