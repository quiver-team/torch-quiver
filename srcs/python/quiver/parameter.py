import torch


class Parameter(torch.nn.parameter.Parameter):
    def __init__(self, shard_tensor):
        super().__init__()
        self.last_input = None
        self.shard_tensor = shard_tensor

    def write_back(self):
        if self.last_input is not None:
            self.shard_tensor[self.last_input] = self.data
        self.last_input = None