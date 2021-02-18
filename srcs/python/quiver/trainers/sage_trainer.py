import torch.nn.functional as F
import torch


class SAGE_trainer():
    def __init__(self, model, device, x, y):
        # super().__init__(model, device)
        self.model = model
        self.model.train()
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        self.x = x
        self.y = y

    def train(self, batch):
        batch_size, n_id, adjs = batch
        adjs = [adj.to(self.device) for adj in adjs]
        self.optimizer.zero_grad()
        out = self.model(self.x[n_id], adjs)
        loss = F.nll_loss(out, self.y[n_id[:batch_size]])
        loss.backward()
        self.optimizer.step()
