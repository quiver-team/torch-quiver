import torch


class Optimizer(torch.optim.Optimizer):
  def __init__(self, model, optimizer) -> None:
      super().__init__()
      self.optimizer_ = optimizer
      self.model_ = model

  def step(self, closure):
      ret = self.optimizer_.step(closure)
      self.model_.emb.write_back()
      return ret
