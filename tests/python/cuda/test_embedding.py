from typing import List
from quiver.utils import reindex_feature, CSRTopo
from quiver.shard_tensor import ShardTensor, ShardTensorConfig, Topo
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import nn
from lib2to3.pgen2.token import OP
from operator import mod
import os
from tkinter.messagebox import NO

import torch
from torch import device, nn, optim
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_quiver as torch_qv
# from quiver import Embedding


device_list = [0, 1]
n_embedding = 4
d_embedding = 4
batch_size = 8


class Embedding(nn.Module):
    def __init__(self, n_embeddings: int, d_embeddings: int, rank: int,
                 device_list: List[int]):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.d_embeddings = d_embeddings
        self.rank = rank
        self.device_list = device_list
        self.ipc_handle_ = None
        self.last_input = None

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
        self.weight.data = self.shard_tensor[input]
        self.last_input = input
        idx = torch.arange(0, len(input)).to(self.rank)
        return F.embedding(idx, self.weight)

    def write_back(self):
        if self.last_input is not None:
            self.shard_tensor[self.last_input] = self.weight.data
        self.last_input = None

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


class Model(nn.Module):
  def __init__(self, n_emb, d_emb, rank, device_list, embedding=None) -> None:
    super().__init__()
    if embedding is None:
      self.emb = Embedding(n_emb, d_emb, rank, device_list)
    else:
      self.emb = embedding
    self.mlp = nn.Linear(d_emb, 1)

  def forward(self, idx):
    embs = self.emb(idx)
    y = self.mlp(embs)
    return y


def simple_test():
  """
  Test basic embedding lookup
  """
  rank = 1
  device = torch.device('cuda', rank) if torch.cuda.is_available() else 'cpu'
  model = Model(n_embedding, d_embedding, rank, device_list).to(device)
  with torch.no_grad():
    x = torch.randint(0, n_embedding, (batch_size,), dtype=torch.long)
    y_ = model(x)
    print(y_)


def simple_bp_test():
  """
  Test embedding lookup with backpropagation
  """
  rank = 0
  device = torch.device('cuda', rank) if torch.cuda.is_available() else 'cpu'
  model = Model(n_embedding, d_embedding, rank, device_list).to(device)
  optimizer = optim.Adam(model.parameters())

  from quiver import Optimizer
  optimizer = Optimizer(model, optimizer)  # Wrap optimizer

  criterion = nn.MSELoss()
  for i in range(2):
    print("-"*32)
    print("Epoch", i)

    x = torch.arange(0, n_embedding, dtype=torch.long)
    y = torch.randn((n_embedding,),).to(device)
    y_ = model(x).squeeze()

    optimizer.zero_grad()
    print("Before training")
    print(model.emb.weight)
    loss = criterion(y_, y)
    loss.backward()
    optimizer.step()
    print("After training")
    print(model.emb.weight)


def mp_test(rank: int, world_size: int, embedding: Embedding):
  """
  Test embedding lookup with multiprocess
  """
  torch.cuda.set_device(rank)
  dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

  device = torch.device('cuda', rank) if torch.cuda.is_available() else 'cpu'

  model = Model(n_embedding, d_embedding, rank,
                device_list, embedding).to(device)
  model = DistributedDataParallel(model, device_ids=[rank])

  with torch.no_grad():
    # x = torch.randint(0, n_embedding, (batch_size,), dtype=torch.long)
    x = torch.arange(n_embedding, dtype=torch.long)
    y_ = model(x)
    print(y_)

  dist.destroy_process_group()


if __name__ == '__main__':
  torch_qv.init_p2p(device_list)

  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '39871'

  n_devices = len(device_list)

  embedding = Embedding(n_embedding, d_embedding, 0, device_list)
  # simple_test()
  # mp.spawn(mp_test, (n_devices, embedding), n_devices)
  simple_bp_test()
