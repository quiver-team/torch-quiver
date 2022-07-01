import torch
from quiver.shard_tensor import ShardTensor, ShardTensorConfig, Topo
from quiver.utils import reindex_feature, CSRTopo, parse_size
from typing import List
import numpy as np
from torch._C import device

__all__ = ["Feature", "DistFeature", "PartitionInfo", "DeviceConfig"]


class DeviceConfig:
    def __init__(self, gpu_parts, cpu_part):
        self.gpu_parts = gpu_parts
        self.cpu_part = cpu_part


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
    def __init__(self,
                 rank: int,
                 device_list: List[int],
                 device_cache_size: int = 0,
                 cache_policy: str = 'device_replicate',
                 csr_topo: CSRTopo = None):
        assert cache_policy in [
            "device_replicate", "p2p_clique_replicate"
        ], f"Feature cache_policy should be one of [device_replicate, p2p_clique_replicate]"
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
        self.mmap_handle_ = None
        self.disk_map = None
        assert self.clique_device_symmetry_check(
        ), f"\n{self.topo.info()}\nDifferent p2p clique size NOT equal"

    def clique_device_symmetry_check(self):
        if self.cache_policy == "device_replicate":
            return True
        print(
            "WARNING: You are using p2p_clique_replicate mode, MAKE SURE you have called quiver.init_p2p() to enable p2p access"
        )
        if len(self.topo.p2pClique2Device.get(1, [])) == 0:
            return True
        if len(self.topo.p2pClique2Device.get(1, [])) == len(
                self.topo.p2pClique2Device[0]):
            return True
        return False
  
    def cal_size(self, cpu_tensor: torch.Tensor, cache_memory_budget: int):
        element_size = cpu_tensor.shape[1] * cpu_tensor.element_size()
        cache_size = cache_memory_budget // element_size
        return cache_size

    def partition(self, cpu_tensor: torch.Tensor, cache_memory_budget: int):

        cache_size = self.cal_size(cpu_tensor, cache_memory_budget)
        return [cpu_tensor[:cache_size], cpu_tensor[cache_size:]]

    def set_mmap_file(self, path, disk_map):
        self.lazy_init_from_ipc_handle()
        self.mmap_handle_ = np.load(path, mmap_mode='r')
        self.disk_map = disk_map.to(self.rank)

    def read_mmap(self, ids):
        ids = ids.cpu().numpy()
        res = torch.from_numpy(self.mmap_handle_[ids])
        res = res.to(device=self.rank, dtype=torch.float32)
        return res

    def from_mmap(self, np_array, device_config):
        """Create quiver.Feature from a mmap numpy array and partition config

        Args:
            np_array (numpy.ndarray): mmap numpy array
            device_config (quiver.feature.DeviceConfig): device partitionconfig
        """
        assert len(device_config.gpu_parts) == len(self.device_list)
        if self.cache_policy == "device_replicate":
            for device in self.device_list:
                if isinstance(device_config.gpu_parts[device], torch.Tensor):
                    if np_array is None:
                        cache_part = device_config.gpu_parts[device].to(dtype=torch.float32)
                    else:
                        cache_ids = device_config.gpu_parts[device].numpy()
                        cache_part = torch.from_numpy(
                            np_array[cache_ids]).to(dtype=torch.float32)
                elif isinstance(device_config.gpu_parts[device], str):
                    cache_part = torch.load(device_config.gpu_parts[device])
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                shard_tensor.append(cache_part, device)
                self.device_tensor_list[device] = shard_tensor
                del cache_part

        else:
            clique0_device_list = self.topo.p2pClique2Device.get(0, [])
            clique1_device_list = self.topo.p2pClique2Device.get(1, [])

            if len(clique0_device_list) > 0:
                print(
                    f"LOG>>> GPU {clique0_device_list} belong to the same NUMA Domain"
                )
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                for idx, device in enumerate(clique0_device_list):
                    if isinstance(device_config.gpu_parts[device],
                                  torch.Tensor):
                        if np_array is None:
                            cache_part = device_config.gpu_parts[device].to(dtype=torch.float32)
                        else:
                            cache_ids = device_config.gpu_parts[device].numpy()
                            cache_part = torch.from_numpy(
                                np_array[cache_ids]).to(dtype=torch.float32)
                    elif isinstance(device_config.gpu_parts[device], str):
                        cache_part = torch.load(
                            device_config.gpu_parts[device])
                    shard_tensor.append(cache_part, device)
                    self.device_tensor_list[device] = shard_tensor
                    del cache_part

                self.clique_tensor_list[0] = shard_tensor

            if len(clique1_device_list) > 0:
                print(
                    f"LOG>>> GPU {clique1_device_list} belong to the same NUMA Domain"
                )
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                for idx, device in enumerate(clique1_device_list):
                    if isinstance(device_config.gpu_parts[device],
                                  torch.Tensor):
                        if np_array is None:
                            cache_part = device_config.gpu_parts[device].to(dtype=torch.float32)
                        else:
                            cache_ids = device_config.gpu_parts[device].numpy()
                            cache_part = torch.from_numpy(
                                np_array[cache_ids]).to(dtype=torch.float32)
                    elif isinstance(device_config.gpu_parts[device], str):
                        cache_part = torch.load(
                            device_config.gpu_parts[device])
                    shard_tensor.append(cache_part, device)
                    self.device_tensor_list[device] = shard_tensor
                    del cache_part

                self.clique_tensor_list[1] = shard_tensor

        # 构建CPU Tensor
        if isinstance(device_config.cpu_part, torch.Tensor):
            if np_array is None:
                self.cpu_part = device_config.cpu_part
            else:
                cache_ids = device_config.cpu_part.numpy()
                self.cpu_part = torch.from_numpy(
                    np_array[cache_ids]).to(dtype=torch.float32)
        elif isinstance(device_config.cpu_part, str):
            self.cpu_part = torch.load(device_config.cpu_part)
        if self.cpu_part.numel() > 0:
            if self.cache_policy == "device_replicate":
                shard_tensor = self.device_tensor_list.get(
                    self.rank, None) or ShardTensor(self.rank,
                                                    ShardTensorConfig({}))
                shard_tensor.append(self.cpu_part, -1)
                self.device_tensor_list[self.rank] = shard_tensor
            else:
                clique_id = self.topo.get_clique_id(self.rank)
                shard_tensor = self.clique_tensor_list.get(
                    clique_id, None) or ShardTensor(self.rank,
                                                    ShardTensorConfig({}))
                shard_tensor.append(self.cpu_part, -1)
                self.clique_tensor_list[clique_id] = shard_tensor

    def from_cpu_tensor(self, cpu_tensor: torch.Tensor):
        """Create quiver.Feature from a pytorh cpu float tensor

        Args:
            cpu_tensor (torch.FloatTensor): input cpu tensor
        """
        if self.cache_policy == "device_replicate":
            cache_memory_budget = parse_size(self.device_cache_size)
            shuffle_ratio = 0.0
        else:
            cache_memory_budget = parse_size(self.device_cache_size) * len(self.topo.p2pClique2Device[0])
            shuffle_ratio = self.cal_size(
                cpu_tensor, cache_memory_budget) / cpu_tensor.size(0)

        print(
            f"LOG>>> {min(100, int(100 * cache_memory_budget / cpu_tensor.numel() / cpu_tensor.element_size()))}% data cached"
        )
        if self.csr_topo is not None:
            if self.csr_topo.feature_order is None:
                cpu_tensor, self.csr_topo.feature_order = reindex_feature(
                    self.csr_topo, cpu_tensor, shuffle_ratio)
            self.feature_order = self.csr_topo.feature_order.to(self.rank)
        cache_part, self.cpu_part = self.partition(cpu_tensor,
                                                   cache_memory_budget)
        self.cpu_part = self.cpu_part.clone()
        if cache_part.shape[0] > 0 and self.cache_policy == "device_replicate":
            for device in self.device_list:
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                shard_tensor.append(cache_part, device)
                self.device_tensor_list[device] = shard_tensor

        elif cache_part.shape[0] > 0:
            clique0_device_list = self.topo.p2pClique2Device.get(0, [])
            clique1_device_list = self.topo.p2pClique2Device.get(1, [])

            block_size = self.cal_size(
                cpu_tensor,
                cache_memory_budget // len(self.topo.p2pClique2Device[0]))

            if len(clique0_device_list) > 0:
                print(
                    f"LOG>>> GPU {clique0_device_list} belong to the same NUMA Domain"
                )
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                cur_pos = 0
                for idx, device in enumerate(clique0_device_list):
                    if idx == len(clique0_device_list) - 1:
                        shard_tensor.append(cache_part[cur_pos:], device)
                    else:

                        shard_tensor.append(
                            cache_part[cur_pos:cur_pos + block_size], device)
                        cur_pos += block_size

                self.clique_tensor_list[0] = shard_tensor

            if len(clique1_device_list) > 0:
                print(
                    f"LOG>>> GPU {clique1_device_list} belong to the same NUMA Domain"
                )
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                cur_pos = 0
                for idx, device in enumerate(clique1_device_list):
                    if idx == len(clique1_device_list) - 1:
                        shard_tensor.append(cache_part[cur_pos:], device)
                    else:

                        shard_tensor.append(
                            cache_part[cur_pos:cur_pos + block_size], device)
                        cur_pos += block_size

                self.clique_tensor_list[1] = shard_tensor

        # 构建CPU Tensor
        if self.cpu_part.numel() > 0:
            if self.cache_policy == "device_replicate":
                shard_tensor = self.device_tensor_list.get(
                    self.rank, None) or ShardTensor(self.rank,
                                                    ShardTensorConfig({}))
                shard_tensor.append(self.cpu_part, -1)
                self.device_tensor_list[self.rank] = shard_tensor
            else:
                clique_id = self.topo.get_clique_id(self.rank)
                shard_tensor = self.clique_tensor_list.get(
                    clique_id, None) or ShardTensor(self.rank,
                                                    ShardTensorConfig({}))
                shard_tensor.append(self.cpu_part, -1)
                self.clique_tensor_list[clique_id] = shard_tensor

    def set_local_order(self, local_order):
        """ Set local order array for quiver.Feature

        Args:
            local_order (torch.Tensor): Tensor which contains the original indices of the features

        """
        local_range = torch.arange(end=local_order.size(0),
                                   dtype=torch.int64,
                                   device=self.rank)
        self.feature_order = torch.zeros_like(local_range)
        self.feature_order[local_order.to(self.rank)] = local_range

    def __getitem__(self, node_idx: torch.Tensor):
        self.lazy_init_from_ipc_handle()
        node_idx = node_idx.to(self.rank)
        if self.mmap_handle_ is None:
            if self.feature_order is not None:
                node_idx = self.feature_order[node_idx]
            if self.cache_policy == "device_replicate":
                shard_tensor = self.device_tensor_list[self.rank]
                return shard_tensor[node_idx]
            else:
                clique_id = self.topo.get_clique_id(self.rank)
                shard_tensor = self.clique_tensor_list[clique_id]
                return shard_tensor[node_idx]
        else:
            num_nodes = node_idx.size(0)
            disk_index = self.disk_map[node_idx]
            node_range = torch.arange(end=num_nodes,
                                      device=self.rank,
                                      dtype=torch.int64)
            disk_mask = disk_index < 0
            mem_mask = disk_index >= 0
            disk_ids = torch.masked_select(node_idx, disk_mask)
            disk_pos = torch.masked_select(node_range, disk_mask)
            mem_ids = torch.masked_select(node_idx, mem_mask)
            mem_pos = torch.masked_select(node_range, mem_mask)
            local_mem_ids = self.disk_map[mem_ids]
            disk_res = self.read_mmap(disk_ids)
            if self.cache_policy == "device_replicate":
                shard_tensor = self.device_tensor_list[self.rank]
                mem_res = shard_tensor[local_mem_ids]
            else:
                clique_id = self.topo.get_clique_id(self.rank)
                shard_tensor = self.clique_tensor_list[clique_id]
                mem_res = shard_tensor[local_mem_ids]
            res = torch.zeros((num_nodes, self.size(1)), device=self.rank)
            res[disk_pos] = disk_res
            res[mem_pos] = mem_res
            return res

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

    def dim(self):
        """ Get the number of dimensions for quiver.Feature

        Args:
            None

        Returns:
            int: number of dimensions
        """
        return len(self.shape)

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
        self.cpu_part.share_memory_()
        gpu_ipc_handle_dict = {}
        if self.cache_policy == "device_replicate":
            for device in self.device_tensor_list:
                gpu_ipc_handle_dict[device] = self.device_tensor_list[
                    device].share_ipc()[0]
        else:
            for clique_id in self.clique_tensor_list:
                gpu_ipc_handle_dict[clique_id] = self.clique_tensor_list[
                    clique_id].share_ipc()[0]

        return gpu_ipc_handle_dict, self.cpu_part if self.cpu_part.numel() > 0 else None, self.device_list, self.device_cache_size, self.cache_policy, self.csr_topo

    def from_gpu_ipc_handle_dict(self, gpu_ipc_handle_dict, cpu_tensor):
        if self.cache_policy == "device_replicate":
            ipc_handle = gpu_ipc_handle_dict.get(
                self.rank, []), cpu_tensor, ShardTensorConfig({})
            shard_tensor = ShardTensor.new_from_share_ipc(
                ipc_handle, self.rank)
            self.device_tensor_list[self.rank] = shard_tensor

        else:
            clique_id = self.topo.get_clique_id(self.rank)
            ipc_handle = gpu_ipc_handle_dict.get(
                clique_id, []), cpu_tensor, ShardTensorConfig({})
            shard_tensor = ShardTensor.new_from_share_ipc(
                ipc_handle, self.rank)
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
        feature = cls(device_list[0], device_list, device_cache_size,
                      cache_policy)
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


class PartitionInfo:
    """PartitionInfo is the partitioning information of how features are distributed across nodes.
    It is mainly used for distributed feature collection, by DistFeature.

    Args:
        device (int): device for local feature partition
        host (int): host id for current node
        hosts (int): the number of hosts in the cluster
        global2host (torch.Tensor): global feature id to host id mapping
        replicate (torch.Tensor, optional): CSRTopo of the graph for feature reordering
        
    """
    def __init__(self, device, host, hosts, global2host, replicate=None):
        self.global2host = global2host.to(device)
        self.host = host
        self.hosts = hosts
        self.device = device
        self.size = self.global2host.size(0)
        self.replicate = None
        if replicate is not None:
            self.replicate = replicate.to(device)
        self.init_global2local()

    def init_global2local(self):
        total_range = torch.arange(end=self.size,
                                   device=self.device,
                                   dtype=torch.int64)
        self.global2local = torch.arange(end=self.size,
                                         device=self.device,
                                         dtype=torch.int64)
        for host in range(self.hosts):
            mask = self.global2host == host
            host_nodes = torch.masked_select(total_range, mask)
            host_size = host_nodes.size(0)
            if host == self.host:
                local_size = host_size
            host_range = torch.arange(end=host_size,
                                      device=self.device,
                                      dtype=torch.int64)
            self.global2local[host_nodes] = host_range
        if self.replicate is not None:
            self.global2host[self.replicate] = self.host
            replicate_range = torch.arange(start=local_size,
                                           end=local_size +
                                           self.replicate.size(0),
                                           device=self.device,
                                           dtype=torch.int64)
            self.global2local[self.replicate] = replicate_range

    def dispatch(self, ids):
        host_ids = []
        host_orders = []
        ids_range = torch.arange(end=ids.size(0),
                                 dtype=torch.int64,
                                 device=self.device)
        host_index = self.global2host[ids]
        for host in range(self.hosts):
            mask = host_index == host
            host_nodes = torch.masked_select(ids, mask)
            host_order = torch.masked_select(ids_range, mask)
            host_nodes = self.global2local[host_nodes]
            host_ids.append(host_nodes)
            host_orders.append(host_order)
        torch.cuda.current_stream().synchronize()

        return host_ids, host_orders


class DistFeature:
    """DistFeature stores local features and it can fetch remote features by the network.
    Normally, each trainer process holds a DistFeature object. 
    We can create DistFeature by a local feature object, a partition information object and a network communicator.
    After creation, each worker process can collect features just like a local tensor.
    It is a synchronous operation, which means every process should collect features at the same time.
 
    ```python
    >>> info = quiver.feature.PartitionInfo(...)
    >>> comm = quiver.comm.NcclComm(...)
    >>> quiver_feature = quiver.Feature(...)
    >>> dist_feature = quiver.feature.DistFeature(quiver_feature, info, comm)
    >>> features = dist_feature[node_idx]
    ```

    Args:
        feature (Feature): local feature
        info (PartitionInfo): partitioning information across nodes
        comm (quiver.comm.NcclComm): communication topology for distributed features
        
    """
    def __init__(self, feature, info, comm):
        self.feature = feature
        self.info = info
        self.comm = comm

    def __getitem__(self, ids):
        ids = ids.to(self.comm.device)
        host_ids, host_orders = self.info.dispatch(ids)
        host_feats = self.comm.exchange(host_ids, self.feature)
        feats = torch.zeros((ids.size(0), self.feature.size(1)),
                            device=self.comm.device)
        for feat, order in zip(host_feats, host_orders):
            if feat is not None and order is not None:
                feats[order] = feat
        local_ids, local_order = host_ids[self.info.host], host_orders[
            self.info.host]
        feats[local_order] = self.feature[local_ids]
        return feats
