import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv


class Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(Net, self).__init__()
        in_channels = num_node_features
        out_channels = num_classes
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self, x0, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        x3 = F.dropout(x3, p=0.2, training=self.training)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.lin(x)
        return x.log_softmax(dim=-1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def train_m(self, use_norm, loader, w, optimizer, device):
        super().train()
        total_loss = total_examples = 0
        w.turn_on('sample')
        self.set_aggr('add' if use_norm else 'mean')

        for data in loader:
            w.turn_off('sample')
            w.turn_on('train')
            data = data.to(device)
            optimizer.zero_grad()

            if use_norm:
                edge_weight = data.edge_norm * data.edge_weight
                out = self(data.x, data.edge_index, edge_weight)
                loss = F.nll_loss(out, data.y.squeeze_(), reduction='none')
                loss = (loss * data.node_norm)[data.train_mask].sum()
            else:
                out = self(data.x, data.edge_index)
                loss = F.nll_loss(out[data.train_mask],
                                  data.y[data.train_mask].squeeze_())
            w.turn_on('sample')
            w.turn_off('train')
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_nodes
            total_examples += data.num_nodes

        return total_loss / total_examples
