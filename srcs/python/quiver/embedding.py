import torch
from torch import nn, Tensor
import torch.nn.functional as F

from quiver.shard_tensor import ShardTensor, ShardTensorConfig
from quiver.parameter import Parameter
from typing import List, Optional


class Embedding(nn.Module):
    def __init__(self, n_embeddings: int, d_embeddings: int, rank: int,
                 device_list: List[int], weight=None):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.d_embeddings = d_embeddings
        self.rank = rank
        self.device_list = device_list
        self.ipc_handle_ = None

        self.shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
        self.weight = Parameter()  # Placeholder
        self.weight.shard_tensor = self.shard_tensor

        if weight is not None:
            assert self.n_embeddings == len(weight), "numbers of embeddings are not same"
            self._init_data_placement(weight)
        else:
            self._init_create_data_placement()

    def forward(self, input):
        self.lazy_init_from_ipc_handle()

        self.weight.data = self.shard_tensor[input]
        self.weight.last_input = input

        idx = torch.arange(0, len(input)).to(self.rank)
        return F.embedding(idx, self.weight)

    def _init_data_placement(self, weight):
        # Even distribution
        n_shards = len(self.device_list) + 1
        items_per_shard = (self.n_embeddings + n_shards - 1) // n_shards
        for i, device in enumerate(self.device_list):
            self.shard_tensor.append(weight[i * items_per_shard:(i + 1) * items_per_shard], device)

        items_remained = self.n_embeddings - (n_shards - 1) * items_per_shard
        if items_remained > 0:
            self.shard_tensor.append(weight[-items_remained:], -1)

    def _init_create_data_placement(self):
        # Even distribution
        n_shards = len(self.device_list) + 1
        items_per_shard = (self.n_embeddings + n_shards - 1) // n_shards
        for device in self.device_list:
            self._append(items_per_shard, device)

        items_remained = self.n_embeddings - (n_shards - 1) * items_per_shard
        if items_remained > 0:
            self._append(items_remained, -1)

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


class EmbeddingBag(nn.Module):
    def __init__(self, n_embeddings: int, d_embeddings: int, mode: str, rank: int,
                 device_list: List[int], weight=None):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.d_embeddings = d_embeddings
        self.mode = mode
        self.rank = rank
        self.device_list = device_list
        self.ipc_handle_ = None

        self.shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
        self.weight = Parameter()  # Placeholder
        self.weight.shard_tensor = self.shard_tensor

        if weight is not None:
            assert self.n_embeddings == len(weight), "numbers of embeddings are not same"
            self._init_data_placement(weight)
        else:
            self._init_create_data_placement()

    def forward(self, input, offset, per_sample_weights: Optional[Tensor] = None):
        # print(len(input), input)
        # unique_input, idx = input.unique(return_inverse=True)
        # print(unique_input.size(), idx.size(), offset.size())
        # print(idx)
        # print(len(offset), offset)
        unique_input = input
        self.lazy_init_from_ipc_handle()

        # self.weight = nn.Parameter(self.shard_tensor[unique_input])
        self.weight.data = self.shard_tensor[unique_input].requires_grad_()
        self.weight.last_input = unique_input

        idx = torch.arange(0, len(input)).to(self.rank)
        return F.embedding_bag(idx, self.weight, offset, mode=self.mode, sparse=True,
                               per_sample_weights=per_sample_weights)

    def _init_data_placement(self, weight):
        # Even distribution
        # n_shards = len(self.device_list) + 1
        # items_per_shard = (self.n_embeddings + n_shards - 1) // n_shards
        # for i, device in enumerate(self.device_list):
        #     self.shard_tensor.append(weight[i * items_per_shard:(i + 1) * items_per_shard], device)
        #
        # items_remained = self.n_embeddings - (n_shards - 1) * items_per_shard
        # if items_remained > 0:
        #     self.shard_tensor.append(weight[-items_remained:], -1)

        # Non-distribution
        self.shard_tensor.append(weight, self.rank)

    def _init_create_data_placement(self):
        # Even distribution
        n_shards = len(self.device_list) + 1
        items_per_shard = (self.n_embeddings + n_shards - 1) // n_shards
        for device in self.device_list:
            self._append(items_per_shard, device)

        items_remained = self.n_embeddings - (n_shards - 1) * items_per_shard
        if items_remained > 0:
            self._append(items_remained, -1)

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
        return self.shard_tensor.share_ipc()[
                   0], self.n_embeddings, self.d_embeddings, self.mode, self.rank, self.device_list

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
        gpu_ipc_handle, n_embeddings, d_embeddings, mode, rank, device_list = ipc_handle
        feature = cls(n_embeddings, d_embeddings, mode, rank, device_list)
        feature.from_gpu_ipc_handle_dict(gpu_ipc_handle)

        return feature

    @classmethod
    def lazy_from_ipc_handle(cls, ipc_handle):
        gpu_ipc_handle, n_embeddings, d_embeddings, mode, rank, device_list = ipc_handle
        feature = cls(n_embeddings, d_embeddings, mode, rank, device_list)
        feature.ipc_handle = gpu_ipc_handle
        return feature

    def lazy_init_from_ipc_handle(self):
        if self.ipc_handle is None:
            return

        self.rank = torch.cuda.current_device()
        gpu_ipc_handle = self.ipc_handle
        self.from_gpu_ipc_handle_dict(gpu_ipc_handle)

        self.ipc_handle = None
