import os

import torch
from torch import device, nn, optim
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_quiver as torch_qv
from quiver import Embedding

device_list = [0, 1]
n_embedding = 4
d_embedding = 4
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
