import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import os.path as osp
import os

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import GraphSAINTRandomWalkSampler
from torch_geometric.utils import degree
from quiver.profile_utils import StopWatch
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from quiver.trainers.saint_trainer import SAINT_trainer
from quiver.saint_sampler import CudaRWSampler
from tqdm import tqdm
import copy
from torch.nn import Linear as Lin
#from quiver.schedule.throughput import ThroughputStats, SamplerChooser
from torch_geometric.nn import GraphConv

print("loading the data...")
w = StopWatch('main')
# --- here is ogbn ------------------
home = os.getenv('HOME')
data_dir = osp.join(home, '.pyg')
root = osp.join(data_dir, 'data', 'products')
dataset = PygNodePropPredDataset('ogbn-products', root)
data = dataset[0]
split_idx = dataset.get_idx_split()
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[split_idx['train']] = True
data.train_mask = train_mask

valid_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
valid_mask[split_idx['valid']] = True
data.val_mask = valid_mask
data.val_mask = valid_mask

test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask[split_idx['test']] = True
data.test_mask = test_mask
row, col = data.edge_index
data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.
# -----below is flicker-----------
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
# dataset = Flickr(path)
# data = dataset[0]
# row, col = data.edge_index

parser = argparse.ArgumentParser()
parser.add_argument('--use_normalization', type=bool, default=True)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# evaluator = Evaluator(name='ogbn-products')
x = data.x.to(device)
y = data.y.squeeze().to(device)  # [N, 1]

sample_data = copy.copy(data)
sample_data.x = None
sample_data.y = None
print(y)
print(sample_data.y)
w.tick('load data')


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(GAT, self).__init__()

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.lin1 = torch.nn.Linear(in_channels, heads * hidden_channels)
        self.conv2 = GATConv(heads * hidden_channels,
                             hidden_channels,
                             heads=heads)
        self.lin2 = torch.nn.Linear(heads * hidden_channels,
                                    heads * hidden_channels)
        self.conv3 = GATConv(heads * hidden_channels,
                             out_channels,
                             heads=heads,
                             concat=False)
        self.lin3 = torch.nn.Linear(heads * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        x = F.relu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index) + self.lin2(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, edge_index) + self.lin3(x)
        x = F.dropout(x, p=0.3, training=self.training)
        return x.log_softmax(dim=-1)


# num ste p is calculated by (train_idx/size(0)) / v_subgraph =20
loader = CudaRWSampler(data,
                       0,
                       batch_size=20000,
                       walk_length=1,
                       num_steps=5,
                       sample_coverage=100,
                       save_dir=dataset.processed_dir,
                       num_workers=0)
# subgraph_loader = GraphSAINTRandomWalkSampler(data,
#                        batch_size=200000,
#                        walk_length = 1,
#                        num_steps=5,
#                        sample_coverage=0,
#                        save_dir=dataset.processed_dir,
#                        num_workers=1)
w.tick('create train_loader')

model = GAT(in_channels=dataset.num_node_features,
            hidden_channels=256,
            out_channels=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
w.tick('build model')


def train():
    print(args.use_normalization)
    model.train()
    # model.set_aggr('add' if args.use_normalization else 'mean')
    model.set_aggr('mean')
    total_loss = total_examples = 0
    w.turn_on('sample')
    for data, node_idx in loader:
        w.turn_off('sample')
        w.turn_on('train')
        data = data.to(device)
        data.x = x[node_idx]
        data.y = y[node_idx]
        optimizer.zero_grad()
        if args.use_normalization:
            # edge_weight = data.edge_norm * data.edge_weight
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out, data.y.squeeze_(), reduction='none')
            loss = (loss * data.node_norm)[data.train_mask].sum()
        else:
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask],
                              data.y[data.train_mask].squeeze_())
        w.turn_on('sample')
        w.turn_off('train')
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    w.turn_off('sample')
    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()
    model.set_aggr('mean')
    # calculate the size of the node put x[pre:curr]
    for data in subgraph_loader:
        out = model(data.x.to(device), data.edge_index.to(device))
        pred = out.argmax(dim=-1)
        correct = pred.eq(data.y.squeeze_().to(device))
        accs = []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            accs.append(correct[mask].sum().item() / mask.sum().item())
        return accs


# # warm up
# for i in range(1, 11):
#    for data in loader:
#        # do nothing
#        continue

w.tick('start train')
for epoch in range(1, 181):
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f},')
    w.tick('train one epoch')

# accs = test()
# print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {accs[0]:.4f}, '
#       f'Val: {accs[1]:.4f}, Test: {accs[2]:.4f}')
# w.tick('finish')
del w
