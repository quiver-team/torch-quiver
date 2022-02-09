from typing import Optional, Callable

import torch

import quiver.parameter


class Optimizer(torch.optim.Optimizer):
    def __init__(self, parameters, optimizer: torch.optim.Optimizer) -> None:
        super(Optimizer, self).__init__(parameters, optimizer.defaults)
        self.parameters = parameters

    def _sync_grad(self):
        for p in self.parameters:
            if isinstance(p, quiver.parameter.Parameter):
                pass  # Sync grad

    def _sync_weight(self):
        for p in self.parameters:
            if isinstance(p, quiver.parameter.Parameter):
                p.write_back()

    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        self._sync_grad()
        return super(Optimizer, self).step(closure)
