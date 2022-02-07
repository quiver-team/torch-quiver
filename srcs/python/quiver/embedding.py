import torch
from torch import nn

from quiver.shard_tensor import ShardTensor, ShardTensorConfig, Topo
from quiver.utils import reindex_feature, CSRTopo
from typing import List
import numpy as np
from torch._C import device


class Embedding(nn.Module):
    def __init__(self, n_embeddings: int, d_embeddings: int, rank: int,
                 device_list: List[int]):
        super().__init__()
        self.rank = rank
        self.device_list = device_list
        self.topo = Topo(self.device_list)

        self.shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))

        n_shards = len(device_list)+1
        items_per_shard = (n_embeddings+n_shards-1)//n_shards
        for device in self.device_list:
            embedding_weight = torch.randn(items_per_shard, d_embeddings)
            self.shard_tensor.append(embedding_weight, device)
            del embedding_weight

        items_remained = n_embeddings-(n_shards-1)*items_per_shard
        if items_remained>0:
            embedding_weight = torch.randn(items_remained, d_embeddings)
            self.shard_tensor.append(embedding_weight, -1)
            del embedding_weight

    def forward(self, index):
        return self.shard_tensor[index]
