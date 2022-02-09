from typing import Optional, Callable

import torch


class Optimizer(torch.optim.Optimizer):
    def __init__(self, parameters, optimizer: torch.optim.Optimizer) -> None:
        super(Optimizer, self).__init__(parameters, optimizer.defaults)

    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        return super(Optimizer, self).step(closure)
