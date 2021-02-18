import torch.nn.functional as F
import torch


class SAINT_trainer():
    def __init__(self, model, device, use_norm):
        self.model = model
        self.model.train()
        self.device = device
        self.use_norm = use_norm
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        model.set_aggr('add' if self.use_norm else 'mean')

    def train(self, batch):
        self.optimizer.zero_grad()
        batch = batch.to(self.device)
        if self.use_norm:
            edge_weight = batch.edge_norm * batch.edge_weight
            out = self.model(batch.x, batch.edge_index, edge_weight)
            loss = F.nll_loss(out, batch.y.squeeze_(), reduction='none')
            loss = (loss * batch.node_norm)[batch.train_mask].sum()
        else:
            out = self.model(batch.x, batch.edge_index)
            loss = F.nll_loss(out[batch.train_mask],
                              batch.y[batch.train_mask].squeeze_())
        loss.backward()
        self.optimizer.step()
