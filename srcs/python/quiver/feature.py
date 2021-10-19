import torch
from quiver.shard_tensor import ShardTensor, ShardTensorConfig, Topo
from quiver.utils import reindex_feature, CSRTopo
from typing import List

__all__ = ["Feature"]


class Feature(object):
    """Feature partitions data onto different GPUs' memory and CPU memory and does feature collection with high performance.
    You will need to set `device_cache_size` to tell Feature how much data it can cached on GPUs memory. By default, it will partition data by your  `device_cache_size`, if you want to cache hot data, you can pass
    graph topology `csr_topo` so that Feature will reorder all data by nodes' degree which we expect to provide higher cache hit rate and will offer better performance with regard to cache random data.
    
    ```python
    >>> cpu_tensor = torch.load("cpu_tensor.pt")
    >>> feature = Feature(0, device_list=[0, 1], device_cache_size='200M')
    >>> feature.from_cpu_tensor(cpu_tensor)
    >>> choose_idx = torch.randint(0, feature.size(0), 100)
    >>> selected_feature = feature[choose_idx]
    ```
    Args:
        rank (int): device for feature collection kernel to launch
        device_list ([int]): device list for data placement
        device_cache_size (Union[int, str]): cache data size for each device, can be like `0.9M` or `3GB`
        cache_policy (str, optional): cache_policy for hot data, can be `device_replicate` or `p2p_clique_replicate`, choose `p2p_clique_replicate` when you have NVLinks between GPUs, else choose `device_replicate`. (default: `device_replicate`)
        csr_topo (quiver.CSRTopo): CSRTopo of the graph for feature reordering
        
    """
    def __init__(self, rank: int, device_list: List[int], device_cache_size: int=0, cache_policy: str='device_replicate', csr_topo: CSRTopo=None):
        assert cache_policy in ["device_replicate", "p2p_clique_replicate"], f"Feature cache_policy should be one of [device_replicate, p2p_clique_replicate]"
        self.device_cache_size = device_cache_size
        self.cache_policy = cache_policy
        self.device_list = device_list
        self.device_tensor_list = {}
        self.clique_tensor_list = {}
        self.rank = rank            
        self.topo = Topo(self.device_list)
        self.csr_topo = csr_topo
        self.feature_order = None
        self.ipc_handle_ = None
        assert self.clique_device_symmetry_check(), f"\n{self.topo.info()}\nDifferent p2p clique size NOT equal"

    def clique_device_symmetry_check(self):
        if self.cache_policy == "device_replicate":
            return True
        print("WARNING: You are using p2p_clique_replicate mode, MAKE SURE you have called quiver.init_p2p() to enable p2p access")
        if len(self.topo.p2pClique2Device.get(1, [])) == 0:
            return True
        if len(self.topo.p2pClique2Device.get(1, [])) == len(self.topo.p2pClique2Device[0]):
            return True
        return False

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


    def cal_size(self, cpu_tensor: torch.Tensor, cache_memory_budget: int):
        element_size = cpu_tensor.shape[1] * 4
        cache_size = cache_memory_budget // element_size
        return cache_size

    def partition(self, cpu_tensor: torch.Tensor, cache_memory_budget: int):
        
        cache_size = self.cal_size(cpu_tensor, cache_memory_budget)
        return [cpu_tensor[:cache_size], cpu_tensor[cache_size: ]]

    def from_cpu_tensor(self, cpu_tensor: torch.Tensor):
        """Create quiver.Feature from a pytorh cpu float tensor

        Args:
            cpu_tensor (torch.FloatTensor): input cpu tensor
        """
        if self.cache_policy == "device_replicate":
            cache_memory_budget = self.cal_memory_budget_bytes(self.device_cache_size)
            shuffle_ratio = 0.0
        else:
            cache_memory_budget = self.cal_memory_budget_bytes(self.device_cache_size) * len(self.topo.p2pClique2Device[0])
            shuffle_ratio = self.cal_size(cpu_tensor, cache_memory_budget) / cpu_tensor.size(0)
        
        print(f"LOG>>> {min(100, int(100 * cache_memory_budget / cpu_tensor.numel() / 4))}% data cached")
        if self.csr_topo is not None:
            if self.csr_topo.feature_order is None:
                cpu_tensor, self.csr_topo.feature_order = reindex_feature(self.csr_topo, cpu_tensor, shuffle_ratio)
            self.feature_order = self.csr_topo.feature_order.to(self.rank)
        cache_part, self.cpu_part = self.partition(cpu_tensor, cache_memory_budget)
        self.cpu_part = self.cpu_part.clone()
        if cache_part.shape[0] > 0 and self.cache_policy == "device_replicate":
            for device in self.device_list:
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                shard_tensor.append(cache_part, device)
                self.device_tensor_list[device] = shard_tensor

        elif cache_part.shape[0] > 0:
            clique0_device_list = self.topo.p2pClique2Device.get(0, [])
            clique1_device_list = self.topo.p2pClique2Device.get(1, [])

            block_size = self.cal_size(cpu_tensor, cache_memory_budget // len(self.topo.p2pClique2Device[0]))

            if len(clique0_device_list) > 0:
                print(f"LOG>>> GPU {clique0_device_list} belong to the same NUMA Domain")
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                cur_pos = 0
                for idx, device in enumerate(clique0_device_list):
                    if idx == len(clique0_device_list) - 1:
                        shard_tensor.append(cache_part[cur_pos:], device)
                    else:

                        shard_tensor.append(cache_part[cur_pos: cur_pos + block_size], device)
                        cur_pos += block_size
                    
                self.clique_tensor_list[0] = shard_tensor
            
            if len(clique1_device_list) > 0:
                print(f"LOG>>> GPU {clique1_device_list} belong to the same NUMA Domain")
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                cur_pos = 0
                for idx, device in enumerate(clique1_device_list):
                    if idx == len(clique1_device_list) - 1:
                        shard_tensor.append(cache_part[cur_pos:], device)
                    else:

                        shard_tensor.append(cache_part[cur_pos: cur_pos + block_size], device)
                        cur_pos += block_size

                self.clique_tensor_list[1] = shard_tensor

        # 构建CPU Tensor
        if self.cpu_part.numel() > 0:
            if self.cache_policy == "device_replicate":
                shard_tensor = self.device_tensor_list.get(self.rank, None) or ShardTensor(self.rank, ShardTensorConfig({}))
                shard_tensor.append(self.cpu_part, -1)
                self.device_tensor_list[self.rank] = shard_tensor
            else:
                clique_id = self.topo.get_clique_id(self.rank)
                shard_tensor = self.clique_tensor_list.get(clique_id, None) or ShardTensor(self.rank, ShardTensorConfig({}))
                shard_tensor.append(self.cpu_part, -1)
                self.clique_tensor_list[clique_id] = shard_tensor
            
        
    def __getitem__(self, node_idx: torch.Tensor):
        self.lazy_init_from_ipc_handle()
        node_idx = node_idx.to(self.rank)
        if self.feature_order is not None:
            node_idx = self.feature_order[node_idx]
        if self.cache_policy == "device_replicate":
            shard_tensor = self.device_tensor_list[self.rank]
            return shard_tensor[node_idx]
        else:
            clique_id = self.topo.get_clique_id(self.rank)
            shard_tensor = self.clique_tensor_list[clique_id]
            return shard_tensor[node_idx]

    
    def size(self, dim: int):
        """ Get dim size for quiver.Feature

        Args:
            dim (int): dimension 

        Returns:
            int: dimension size for dim
        """
        self.lazy_init_from_ipc_handle()
        if self.cache_policy == "device_replicate":
            shard_tensor = self.device_tensor_list[self.rank]
            return shard_tensor.size(dim)
        else:
            clique_id = self.topo.get_clique_id(self.rank)
            shard_tensor = self.clique_tensor_list[clique_id]
            return shard_tensor.size(dim)

    @property
    def shape(self):
        self.lazy_init_from_ipc_handle()
        if self.cache_policy == "device_replicate":
            shard_tensor = self.device_tensor_list[self.rank]
            return shard_tensor.shape
        else:
            clique_id = self.topo.get_clique_id(self.rank)
            shard_tensor = self.clique_tensor_list[clique_id]
            return shard_tensor.shape
    

    @property
    def ipc_handle(self):
        return self.ipc_handle_
            
    @ipc_handle.setter
    def ipc_handle(self, ipc_handle):
        self.ipc_handle_ = ipc_handle

    def share_ipc(self):
        """Get ipc handle for multiprocessing

        Returns:
            tuples: ipc handles for ShardTensor and torch.Tensor and python native objects
        """
        gpu_ipc_handle_dict = {}
        if self.cache_policy == "device_replicate":
            for device in self.device_tensor_list:
                gpu_ipc_handle_dict[device] = self.device_tensor_list[device].share_ipc()[0]
        else:
            for clique_id in self.clique_tensor_list:
                gpu_ipc_handle_dict[clique_id] = self.clique_tensor_list[clique_id].share_ipc()[0]
        
        return gpu_ipc_handle_dict, self.cpu_part, self.device_list, self.device_cache_size, self.cache_policy, self.csr_topo
    
    def from_gpu_ipc_handle_dict(self, gpu_ipc_handle_dict, cpu_tensor):
        if self.cache_policy == "device_replicate":
            ipc_handle = gpu_ipc_handle_dict.get(self.rank, []), cpu_tensor, ShardTensorConfig({})
            shard_tensor = ShardTensor.new_from_share_ipc(ipc_handle, self.rank)
            self.device_tensor_list[self.rank] = shard_tensor
            
        else:
            clique_id = self.topo.get_clique_id(self.rank)
            ipc_handle = gpu_ipc_handle_dict.get(clique_id, []), cpu_tensor, ShardTensorConfig({})
            shard_tensor = ShardTensor.new_from_share_ipc(ipc_handle, self.rank)
            self.clique_tensor_list[clique_id] = shard_tensor
        
        self.cpu_part = cpu_tensor
        
        
    @classmethod
    def new_from_ipc_handle(cls, rank, ipc_handle):
        """Create from ipc handle

        Args:
            rank (int): device rank for feature collection kernels to launch
            ipc_handle (tuple): ipc handle create from `share_ipc`

        Returns:
            [quiver.Feature]: created quiver.Feature
        """
        gpu_ipc_handle_dict, cpu_part, device_list, device_cache_size, cache_policy, csr_topo = ipc_handle
        feature = cls(rank, device_list, device_cache_size, cache_policy)
        feature.from_gpu_ipc_handle_dict(gpu_ipc_handle_dict, cpu_part)
        if csr_topo is not None:
            feature.feature_order = csr_topo.feature_order.to(rank)
        feature.csr_topo = csr_topo
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