import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import random_walk
from sklearn.linear_model import LogisticRegression

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler as RawNeighborSampler

######################
# Import From Quiver
######################
import quiver
from quiver.pyg import GraphSageSampler

EPS = 1e-15

dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
path = '/data/' + dataset
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

train_idx = torch.arange(data.num_nodes, dtype=torch.long)
#############################
# Original Pyg Code
#############################
# train_loader = NeighborSampler(data.edge_index, sizes=[10, 10], batch_size=256,
#                                shuffle=True, num_nodes=data.num_nodes)
# print(train_idx)
train_loader = torch.utils.data.DataLoader(train_idx,
                                           batch_size=256,
                                           shuffle=True,
                                           drop_last=True)

csr_topo = quiver.CSRTopo(data.edge_index)

quiver_sampler = GraphSageSampler(csr_topo, sizes =[10, 10], device=0)

def sample(edge_index, batch):
    batch = torch.tensor(batch)
    row, col = edge_index[0], edge_index[1]

    # For each node in `batch`, we sample a direct neighbor (as positive
    # example) and a random node (as negative example):
    pos_batch = random_walk(row, col, batch, walk_length=1,
                            coalesced=False)[:, 1]
    
    neg_batch = torch.randint(0, csr_topo.indptr.shape[-1] - 1, (batch.numel(), ),
                                dtype=torch.long)

    batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
    return quiver_sampler.sample(batch)


class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(data.num_node_features, hidden_channels=64, num_layers=2)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
####################
# Original Pyg Code
####################
x, edge_index = data.x.to(device), data.edge_index.to(device)
quiver_feature = quiver.Feature(rank=0, device_list=[0], device_cache_size="15M", cache_policy="device_replicate", csr_topo=csr_topo)
quiver_feature.from_cpu_tensor(data.x)

def train():
    model.train()
    total_loss = 0
    ######################
    # Original Pyg Code
    ######################
    # for batch_size, n_id, adjs in train_loader:
    for seeds in train_loader:
        n_id, batch_size, adjs = sample(data.edge_index, seeds)
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out = model(quiver_feature[n_id], adjs)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss = -pos_loss - neg_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * out.size(0)

    return total_loss / data.num_nodes


@torch.no_grad()
def test():
    model.eval()
    out = model.full_forward(x, edge_index).cpu()

    clf = LogisticRegression()
    clf.fit(out[data.train_mask], data.y[data.train_mask])

    val_acc = clf.score(out[data.val_mask], data.y[data.val_mask])
    test_acc = clf.score(out[data.test_mask], data.y[data.test_mask])

    return val_acc, test_acc


for epoch in range(1, 51):
    loss = train()
    val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')