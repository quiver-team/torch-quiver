import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch_scatter import scatter
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.data import RandomNodeSampler
from quiver.node_sampler import RandomNodeCudaSampler
import os
import os.path as osp
from quiver.profile_utils import StopWatch

w = StopWatch('main')
home = os.getenv('HOME')
data_dir = osp.join(home, '.pyg')
root = osp.join(data_dir, 'data', 'products')
dataset = PygNodePropPredDataset('ogbn-products', root)

splitted_idx = dataset.get_idx_split()
data = dataset[0]
data.node_species = None
data.y = data.y.squeeze()
evaluator = Evaluator('ogbn-products')
# Initialize features of nodes by aggregating edge features.
row, col = data.edge_index

# Set split indices to masks.
for split in ['train', 'valid', 'test']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx[split]] = True
    data[f'{split}_mask'] = mask
w.tick('load data')
train_loader = RandomNodeCudaSampler(data, 5, num_parts=10, shuffle=True)
test_loader = RandomNodeSampler(data, num_parts=5, num_workers=5)
w.tick('create train_loader')


class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super(DeeperGCN, self).__init__()
        self.node_encoder = Linear(data.x.size(-1), hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels,
                           hidden_channels,
                           aggr='softmax_sg',
                           t=1.0,
                           learn_t=True,
                           num_layers=1,
                           norm='batch')
            norm = LayerNorm(hidden_channels, elementwise_affine=False)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv,
                                 norm,
                                 act,
                                 block='res+',
                                 dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

    def forward(self, x, edge_index):
        x = self.node_encoder(x)

        x = self.layers[0].conv(x, edge_index)

        for layer in self.layers[1:]:
            x = layer(x, edge_index)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return torch.log_softmax(x, dim=-1)


device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
model = DeeperGCN(hidden_channels=128, num_layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
# evaluator = Evaluator('ogbn-proteins')
w.tick('build model')


def train(epoch):
    model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training epoch: {epoch:04d}')
    w.turn_on('sample')
    total_loss = total_examples = 0
    for data in train_loader:
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = F.nll_loss(out[data.train_mask],
                          data.y[data.train_mask].squeeze().to(device))
        loss.backward()
        w.turn_off('sample')
        # data = data.to(device)
        w.turn_on('train')
        optimizer.zero_grad()

        optimizer.step()
        w.turn_off('train')
        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)
        w.turn_on('sample')
    w.turn_off('sample')
    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f'Evaluating epoch: {epoch:04d}')

    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)

        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        pbar.update(1)

    pbar.close()

    train_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['rocauc']

    valid_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['rocauc']

    test_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


for epoch in range(1, 2):
    loss = train(epoch)
    # train_rocauc, valid_rocauc, test_rocauc = test()
    # print(f'Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
    #       f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
    w.tick('train one epoch')
    print(f'Loss: {loss:.4f}')
del w
