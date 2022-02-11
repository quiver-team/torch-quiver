import torch


class Parameter(torch.nn.parameter.Parameter):
    def __init__(self):
        super().__init__()
        self.last_input = None
        self.shard_tensor = None

    def write_back(self):
        assert self.shard_tensor is not None, "Shard tensor is not initialized"

        if self.last_input is not None:
            self.shard_tensor.update(self.last_input, self.data)
            # self.shard_tensor[self.last_input] #= self.data
        self.last_input = None
