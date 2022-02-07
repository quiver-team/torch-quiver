import imp


import torch
from torch import nn

from quiver import Embedding


class Model(nn.Module):
  def __init__(self, rank) -> None:
    super().__init__()
    self.emb = Embedding(4, 16, rank, [0, 1])

  def forward(self, idx):
    embs = self.emb(idx)
    return embs


if __name__ == '__main__':
  rank = 0
  x = torch.tensor([2], dtype=torch.long, requires_grad=False)
  model = Model(rank)
  with torch.no_grad():
    y_ = model(x)
    print(y_)
