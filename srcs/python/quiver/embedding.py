import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from quiver.shard_tensor import ShardTensor, ShardTensorConfig, Topo
from quiver.utils import reindex_feature, CSRTopo
from typing import List


class Embedding(nn.Module):
    def __init__(self, n_embeddings: int, d_embeddings: int, rank: int,
                 device_list: List[int]):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.d_embeddings = d_embeddings
        self.rank = rank
        self.device_list = device_list
        self.ipc_handle_ = None

        self.weight = Parameter()  # Placeholder
        self.shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))

        n_shards = len(device_list)+1
        items_per_shard = (n_embeddings+n_shards-1)//n_shards
        for device in self.device_list:
            self._append(items_per_shard, device)

        items_remained = n_embeddings-(n_shards-1)*items_per_shard
        if items_remained > 0:
            self._append(items_remained, -1)

    def forward(self, input):
        self.lazy_init_from_ipc_handle()
        # Modify the ptr of self.weight to fetched data
        self.shard_tensor.get(input, self.weight)
        idx = torch.arange(0, len(input)).to(self.rank)
        return F.embedding(idx, self.weight)

    def _append(self, n_items, device):
        if n_items > 0:
            embedding_weight = torch.randn(n_items, self.d_embeddings)
            self.shard_tensor.append(embedding_weight, device)
            del embedding_weight

    @property
    def ipc_handle(self):
        return self.ipc_handle_

    @ipc_handle.setter
    def ipc_handle(self, ipc_handle):
        self.ipc_handle_ = ipc_handle

    def share_ipc(self):
        """Get ipc handle for multiprocessing

        Returns:
            tuples: ipc handles for ShardTensor and 
        """
        return self.shard_tensor.share_ipc()[0], self.n_embeddings, self.d_embeddings, self.rank, self.device_list

    def from_gpu_ipc_handle_dict(self, gpu_ipc_handle):
        ipc_handle = gpu_ipc_handle, None, ShardTensorConfig({})
        self.shard_tensor = ShardTensor.new_from_share_ipc(
            ipc_handle, self.rank)

    @classmethod
    def new_from_ipc_handle(cls, rank, ipc_handle):
        """Create from ipc handle

        Args:
            rank (int): device rank for embedding collection kernels to launch
            ipc_handle (tuple): ipc handle create from `share_ipc`

        Returns:
            [quiver.Embedding]: created quiver.Embedding
        """
        gpu_ipc_handle, n_embeddings, d_embeddings, rank, device_list = ipc_handle
        feature = cls(n_embeddings, d_embeddings, rank, device_list)
        feature.from_gpu_ipc_handle_dict(gpu_ipc_handle)

        return feature

    @classmethod
    def lazy_from_ipc_handle(cls, ipc_handle):
        gpu_ipc_handle, n_embeddings, d_embeddings, rank, device_list = ipc_handle
        feature = cls(n_embeddings, d_embeddings, rank, device_list)
        feature.ipc_handle = gpu_ipc_handle
        return feature

    def lazy_init_from_ipc_handle(self):
        if self.ipc_handle is None:
            return

        self.rank = torch.cuda.current_device()
        gpu_ipc_handle = self.ipc_handle
        self.from_gpu_ipc_handle_dict(gpu_ipc_handle)

        self.ipc_handle = None
