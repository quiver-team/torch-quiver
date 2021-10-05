import os.path as osp
import os

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Flickr
from torch_geometric.utils import degree
from quiver.profile_utils import StopWatch
# from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from quiver.trainers.saint_trainer import SAINT_trainer
from quiver.models.saint_model import Net
from quiver.saint_sampler import CudaRWSampler
from quiver.saint_sampler import quiverRWSampler
#from quiver.schedule.throughput import ThroughputStats, SamplerChooser
from torch_geometric.nn import GraphConv
from torch_geometric.data import Data
import copy

print("loading the data...")
w = StopWatch('main')
# --- here is ogbn ------------------
# home = os.getenv('HOME')
# data_dir = osp.join(home, '.pyg')
# root = osp.join(data_dir, 'data', 'products')
# dataset = PygNodePropPredDataset('ogbn-products', root)
# data = dataset[0]
# split_idx = dataset.get_idx_split()
# valid_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
# valid_mask[split_idx['valid']] = True
# data.val_mask = valid_mask
# data.val_mask = valid_mask
#
# test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
# test_mask[split_idx['test']] = True
# data.test_mask= test_mask
# row, col = data.edge_index
# evaluator = Evaluator(name='ogbn-products')
# -----below is flicker-----------
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
# dataset = Flickr(path)
# data = dataset[0]
# row, col = data.edge_index

# -----below is flicker-----------


class ProductsDataset:
    def __init__(self, train_idx, edge_index, x, y, f, c):
        self.train_idx = train_idx
        self.edge_index = edge_index
        self.x = x
        self.y = y
        self.num_features = f
        self.num_classes = c
        row, col = edge_index
        self.num_edges = row.size(0)
        self.num_nodes = 2500000

    @property
    def keys(self):
        r"""Returns all names of graph attributes."""
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys

    def __iter__(self):
        r"""Iterates over all present attributes in the data, yielding their
        attribute names and content."""
        for key in sorted(self.keys):
            yield key, self[key]

    def __contains__(self, key):
        r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def __call__(self, *keys):
        r"""Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes."""
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    def __contains__(self, key):
        r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)


root = '../products.pt'
dataset = torch.load(root)
train_idx = dataset.train_idx
data = dataset

train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
row, col = data.edge_index
dataset.processed_dir = "./"
data.train_mask = train_mask

parser = argparse.ArgumentParser()
parser.add_argument('--use_normalization', type=bool, default=True)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = data.x.to(device)
y = data.y.squeeze().to(device)  # [N, 1]

sample_data = copy.copy(data)
sample_data.x = None
sample_data.y = None

w.tick('load data')
# num ste p is calculated by (train_idx/size(0)) / v_subgraph =20
loader = CudaRWSampler(sample_data,
                       0,
                       batch_size=20000,
                       walk_length=1,
                       num_steps=5,
                       sample_coverage=100,
                       save_dir=dataset.processed_dir,
                       num_workers=0)
# subgraph_loader = quiverRWSampler(data,
#                        batch_size=200000,
#                        walk_length = 1,
#                        num_steps=1,
#                        sample_coverage=0,
#                        save_dir=dataset.processed_dir,
#                        num_workers=4)
w.tick('create train_loader')
print(dataset.num_features)

model = Net(hidden_channels=256,
            num_node_features=dataset.num_features,
            num_classes=dataset.num_classes).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
w.tick('build model')

# trainer = SAINT_trainer(model, device, args.use_normalization)
# chooser = SamplerChooser(trainer)
# train_loader = chooser.choose_sampler((loader, loader))
# if isinstance(train_loader, GraphSAINTRandomWalkSampler):
#     print('choose cpu sampler')
# else:
#     print('choose cuda sampler')
# w.tick('choose sampler')


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


# @torch.no_grad()
# def test():
#     model.eval()
#     model.set_aggr('mean')
#     # calculate the size of the node put x[pre:curr]
#     for data in subgraph_loader:
#         out = model(data.x.to(device), data.edge_index.to(device))
#         pred = out.argmax(dim=-1)
#         correct = pred.eq(data.y.squeeze_().to(device))
#         accs = []
#         for _, mask in data('train_mask', 'val_mask', 'test_mask'):
#             accs.append(correct[mask].sum().item() / mask.sum().item())
#         return accs

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
