import os.path as osp
import os

import argparse
import torch
import torch.nn.functional as F
#from torch_geometric.datasets import Flickr
from torch_geometric.data import GraphSAINTRandomWalkSampler
from torch_geometric.utils import degree
from quiver.profile_utils import StopWatch
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from quiver.trainers.saint_trainer import Net


print("loading the data...")
w = StopWatch('main')
home = os.getenv('HOME')
data_dir = osp.join(home, '.pyg')
root = osp.join(data_dir, 'data', 'products')
dataset = PygNodePropPredDataset('ogbn-products', root)
data = dataset[0]

split_idx = dataset.get_idx_split()
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[split_idx['train']]= True
data.train_mask=train_mask

row, col = data.edge_index
data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.

parser = argparse.ArgumentParser()
parser.add_argument('--use_normalization', action='store_true')
args = parser.parse_args()
w.tick('load data')

loader = GraphSAINTRandomWalkSampler(data, batch_size=1000, walk_length=2,
                                     num_steps=1, sample_coverage=5,
                                     save_dir=dataset.processed_dir,
                                     num_workers=4)

w.tick('create train_loader')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(hidden_channels=256, num_node_features=dataset.num_node_features,
            num_classes=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    model.set_aggr('add' if args.use_normalization else 'mean')

    total_loss = total_examples = 0
    w.turn_on('sample')
    for data in loader:
        w.turn_off('sample')
        data = data.to(device)
        optimizer.zero_grad()

        if args.use_normalization:
            edge_weight = data.edge_norm * data.edge_weight
            out = model(data.x, data.edge_index, edge_weight)
            loss = F.nll_loss(out, data.y.squeeze_(), reduction='none')
            loss = (loss * data.node_norm)[data.train_mask].sum()
            print('loss:', loss.shape)
            print('out:', out.shape)
            print('data.y',data.y.shape)
        else:
            print(data.x.shape)
            print(data.edge_index.shape)
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].squeeze_())
        w.turn_on('sample')
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()
    model.set_aggr('mean')

    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y.to(device))

    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(correct[mask].sum().item() / mask.sum().item())
    return accs


for epoch in range(1, 51):
    loss = train()
    #accs = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, ')
          # f'Train: {accs[0]:.4f}, '
          # f'Val: {accs[1]:.4f}, Test: {accs[2]:.4f}')

w.tick('finish')
del w