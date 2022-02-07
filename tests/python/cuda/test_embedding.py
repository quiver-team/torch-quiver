import torch
from torch import nn

import torch_quiver as torch_qv
from quiver import Embedding


device_list = [0, 1]
n_embedding = 4
d_embedding = 16
batch_size = 16


class Model(nn.Module):
  def __init__(self, n_emb, d_emb, rank, device_list) -> None:
    super().__init__()
    self.emb = Embedding(n_emb, d_emb, rank, device_list)
    self.mlp = nn.Linear(d_emb, 1).to(rank)

  def forward(self, idx):
    embs = self.emb(idx)
    return embs


if __name__ == '__main__':
  torch_qv.init_p2p([0, 1])
  rank = 0
  model = Model(n_embedding, d_embedding, rank, device_list)
  with torch.no_grad():
    x = torch.randint(0, n_embedding, (batch_size,), dtype=torch.long)
    y_ = model(x)
    print(y_)
