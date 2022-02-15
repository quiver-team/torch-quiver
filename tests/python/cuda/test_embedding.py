import os
import time

import torch
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_quiver as torch_qv
from quiver import Embedding, EmbeddingBag

device_list = [0, 1]
n_embedding = 6
d_embedding = 2
batch_size = 8


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


class ModelBag(nn.Module):
    def __init__(self, n_emb, d_emb, mode, rank, device_list, embedding_bag=None) -> None:
        super().__init__()
        if embedding_bag is None:
            self.emb = EmbeddingBag(n_emb, d_emb, mode, rank, device_list)
        else:
            self.emb = embedding_bag
        self.mlp = nn.Linear(d_emb, 1)

    def forward(self, index, offset):
        embs = self.emb(index, offset)
        y = self.mlp(embs)
        return y


class ModelPtBag(nn.Module):
    def __init__(self, n_emb, d_emb, mode, embedding_bag=None) -> None:
        super().__init__()
        if embedding_bag is None:
            self.emb = nn.EmbeddingBag(n_emb, d_emb, mode=mode, sparse=True)
        else:
            self.emb = embedding_bag
        self.mlp = nn.Linear(d_emb, 1)

    def forward(self, index, offset):
        embs = self.emb(index, offset)
        y = self.mlp(embs)
        return y


def simple_test():
    """
    Test basic embedding lookup
    """
    rank = 1
    device = torch.device('cuda', rank) if torch.cuda.is_available() else 'cpu'

    weight = torch.randn(n_embedding, d_embedding)
    quiver_emb = Embedding(n_embedding, d_embedding, rank, device_list, weight).to(device)
    torch_emb = nn.Embedding(n_embedding, d_embedding, _weight=weight).to(device)

    x = torch.randint(n_embedding, (batch_size,), dtype=torch.long)
    print("Torch embedding")
    print(torch_emb(x.to(device)))
    print("Quiver embedding")
    print(quiver_emb(x))


def simple_bp_test():
    """
    Test embedding lookup with backpropagation
    """
    rank = 0
    device = torch.device('cuda', rank) if torch.cuda.is_available() else 'cpu'
    model = Model(n_embedding, d_embedding, rank, device_list).to(device)
    optimizer = optim.Adam(model.parameters())

    from quiver import SynchronousOptimizer
    optimizer = SynchronousOptimizer(model.parameters(), optimizer)  # Wrap optimizer

    criterion = nn.MSELoss()
    for i in range(2):
        print("-" * 32)
        print("Epoch", i)

        x = torch.arange(0, n_embedding, dtype=torch.long)
        y = torch.randn((n_embedding,), ).to(device)
        y_ = model(x).squeeze()

        optimizer.zero_grad()
        print("Before training")
        print(model.emb.weight)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()
        # model.emb.shard_tensor[x]#=model.emb.weight.data
        print("After training")
        print(model.emb.weight)


def simple_bag_test():
    rank = 0
    mode = "sum"
    device = torch.device('cuda', rank) if torch.cuda.is_available() else 'cpu'

    idx = torch.randint(n_embedding, (batch_size,), dtype=torch.long)
    offset = torch.tensor([0, batch_size // 2]).to(rank)
    weight = torch.randn(n_embedding, d_embedding)

    torch_embb = nn.EmbeddingBag(n_embedding, d_embedding, mode=mode, _weight=weight).to(device)
    quiver_embb = EmbeddingBag(n_embedding, d_embedding, mode, rank, device_list, weight).to(device)

    print("Torch embedding bag")
    print(torch_embb(idx.to(device), offset.to(device)))
    print("Quiver embedding bag")
    print(quiver_embb(idx, offset))


def simple_bp_bag_test():
    """
    Test embedding bag lookup with backpropagation
    """
    rank = 0
    device = torch.device('cuda', rank) if torch.cuda.is_available() else 'cpu'
    model = ModelBag(n_embedding, d_embedding, "sum", rank, device_list).to(device)
    pt_model = ModelPtBag(n_embedding, d_embedding, "sum").to(device)
    # pt_model.mlp.weight.data=model.mlp.weight.data
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    pt_optimizer = optim.SGD(pt_model.parameters(), lr=0.1)

    from quiver import SynchronousOptimizer
    optimizer = SynchronousOptimizer(model.parameters(), optimizer)  # Wrap optimizer

    criterion = nn.MSELoss()
    for i in range(2):
        print("-" * 32)
        print("Epoch", i)

        if i == 0:
            x = torch.tensor([0, 2, 1, 3, 2, 3, 4, 5], dtype=torch.long).to(device)
        else:
            x = torch.tensor([0, 2, 1, 3, 5], dtype=torch.long).to(device)
        o = torch.tensor([0, 4]).to(device)
        y = torch.randn((len(o),), ).to(device)
        y_ = model(x, o).squeeze()
        # pt_y_ = pt_model(x, o).squeeze()

        optimizer.zero_grad()
        pt_optimizer.zero_grad()
        print("Before training")
        print(model.emb.weight)
        loss = criterion(y_, y)
        # pt_loss = criterion(pt_y_, y)
        loss.backward()
        # pt_loss.backward()
        optimizer.step()
        # pt_optimizer.step()
        # model.emb.shard_tensor[x]#=model.emb.weight.data
        print("After training")
        print(model.emb.weight)


def pt_dynamic_bag_test():
    rank = 0
    device = torch.device('cuda', rank) if torch.cuda.is_available() else 'cpu'
    pt_model = ModelPtBag(n_embedding, d_embedding, "sum").to(device)
    pt_optimizer = optim.SGD(pt_model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    x = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long).to(device)
    o = torch.tensor([0, 2]).to(device)
    y = torch.randn((len(o),), ).to(device)
    pt_y_ = pt_model(x, o).squeeze()

    pt_optimizer.zero_grad()
    pt_loss = criterion(pt_y_, y)
    pt_loss.backward()
    pt_optimizer.step()

    x = torch.tensor([0, 1, 2], dtype=torch.long).to(device)
    pt_model.emb.weight.data = torch.randn(4, d_embedding, requires_grad=True).to(device)
    pt_y_ = pt_model(x, o).squeeze()

    pt_optimizer.zero_grad()
    pt_loss = criterion(pt_y_, y)
    pt_loss.backward()
    pt_optimizer.step()


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
    # simple_bp_test()
    simple_bp_bag_test()
