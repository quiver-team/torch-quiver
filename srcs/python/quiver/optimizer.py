import torch

import quiver.parameter


class _SynchronousOptimizer(torch.optim.Optimizer):
    def __init__(self, parameters, param_groups) -> None:
        super(self.__class__, self).__init__(param_groups)
        self.parameters = list(parameters)

    def _sync_grad(self):
        for p in self.parameters:
            if isinstance(p, quiver.parameter.Parameter):
                pass  # Sync grad

    def _sync_weight(self):
        for p in self.parameters:
            if isinstance(p, quiver.parameter.Parameter):
                p.write_back()

    def step(self, closure=None):
        # self._sync_grad()
        ret = super(self.__class__, self).step(closure)
        self._sync_weight()
        return ret


def SynchronousOptimizer(parameters, optimizer: torch.optim.Optimizer):
    clazz = type(optimizer.__class__.__name__, (optimizer.__class__,),
                 dict(_SynchronousOptimizer.__dict__))
    return clazz(parameters, optimizer.param_groups)
