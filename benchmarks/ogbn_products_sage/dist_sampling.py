#!/usr/bin/env python3
# modified from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_products_sage.py
# Reaches around 0.7870 ± 0.0036 test accuracy.

import argparse
import os
import os.path as osp
import time

import kungfu.torch as kf
from kungfu.python import current_cluster_size, current_rank, run_barrier
import torch
import torch.nn.functional as F
from torch.distributed import rpc
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from quiver.profile_utils import StopWatch
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

p = argparse.ArgumentParser(description='')
p.add_argument('--cuda', type=bool, default=True, help='cuda')
p.add_argument('--ws', type=int, default=4, help='world size')
p.add_argument('--rank', type=int, default=0, help='rank')
p.add_argument('--runs', type=int, default=1, help='number of runs')
p.add_argument('--epochs', type=int, default=1, help='number of epochs')
args = p.parse_args()
torch.cuda.set_device(args.rank)
if args.cuda:
    import quiver.dist_cuda_sampler as dist
else:
    import quiver.dist_cpu_sampler as dist


def node2rank(nodes):
    ranks = torch.fmod(nodes, args.ws)
    return ranks


def local2global(nodes):
    return nodes


def global2local(nodes):
    return nodes


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

print('loading ... ')
w = StopWatch('main')
home = os.getenv('HOME')
data_dir = osp.join(home, '.pyg')
root = osp.join(data_dir, 'data', 'products')
dataset = PygNodePropPredDataset('ogbn-products', root)
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-products')
data = dataset[0]
args.rank = current_rank()
args.ws = current_cluster_size()
dev = torch.device(0)
cpu = torch.device('cpu')
x = data.x.share_memory_()  # [N, 100]
y = data.y.squeeze().share_memory_()  # [N, 1]

train_idx = split_idx['train']

w.tick('load data')

comm = dist.Comm(args.rank, args.ws)


def node_f(nodes, is_feature):
    if is_feature:
        return x[nodes].to(cpu)
    else:
        return y[nodes].to(cpu)


train_loader = dist.SyncDistNeighborSampler(
    comm, (int(data.edge_index.max() + 1), data.edge_index,
           (x, y), local2global, global2local, node2rank),
    train_idx, [15, 10, 5],
    0,
    node_f,
    batch_size=1024)


def sample_cuda(nodes, size):
    torch.cuda.set_device(0)
    nodes = train_loader.global2local(nodes)
    neighbors, counts = train_loader.quiver.sample_neighbor(0, nodes, size)
    neighbors = train_loader.local2global(neighbors)

    return neighbors, counts


def sample_cpu(nodes, size):
    nodes = train_loader.global2local(nodes)
    neighbors, counts = train_loader.quiver.sample_neighbor(nodes, size)
    neighbors = train_loader.local2global(neighbors)

    return neighbors, counts


if args.cuda:
    dist.sample_neighbor = sample_cuda
else:
    dist.sample_neighbor = sample_cpu

w.tick('create train_loader')

rpc.init_rpc(f"worker{args.rank}",
             rank=args.rank,
             world_size=args.ws,
             rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                 num_worker_threads=8, rpc_timeout=20))


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=3)
model = model.to(device)

w.tick('build model')


def train(epoch):
    # w1 = StopWatch('train loop')
    model.train()
    # w1.tick('set mode to train')

    # pbar = tqdm(total=train_idx.size(0))
    # pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    w.turn_on('sample')
    t0 = time.time()
    torch.set_num_threads(5)
    samples = [sample for sample in train_loader]
    run_barrier()
    for n_id, batch_size, adjs in samples:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        # w1.tick('prepro')
        w.turn_off('sample')
        w.turn_on('train')
        adjs = [adj.to(device) for adj in adjs]
        feature = x[n_id].to(device)
        label = y[n_id[:batch_size]].to(device)
        # feature, order0 = feature
        # label, order1 = label
        # feature = torch.cat([f.to(device) for f in feature])
        # label = torch.cat([l.to(device) for l in label])
        # origin_feature = torch.empty_like(feature)
        # origin_label = torch.empty_like(label)
        # origin_feature[order0] = feature
        # origin_label[order1] = label

        optimizer.zero_grad()
        out = model(feature, adjs)
        loss = F.nll_loss(out, label)
        loss.backward()
        optimizer.step()
        # w1.tick('train')

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(label).sum())
        # pbar.update(batch_size)
        torch.cuda.synchronize(0)
        w.turn_on('sample')
        w.turn_off('train')
        # if args.rank == 0:
        #     print(f'one step took {time.time() - t0}')
        t0 = time.time()

    # pbar.close()
    w.turn_off('sample')
    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    # del w1
    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc


model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
test_accs = []
for run in range(1, 1 + args.runs):
    print('')
    print(f'Run {run:02d}:')
    print('')

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    optimizer = kf.optimizers.SynchronousSGDOptimizer(
        optimizer, named_parameters=model.named_parameters())
    # Broadcast parameters from rank 0 to all other processes.
    kf.broadcast_parameters(model.state_dict())

    best_val_acc = final_test_acc = 0.0
    w.tick('?')
    for epoch in range(1, 1 + args.epochs):
        loss, acc = train(epoch)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        w.tick('train one epoch')

        if epoch > 5:
            train_acc, val_acc, test_acc = test()
            print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
            test_accs.append(final_test_acc)

if test_accs:
    test_acc = torch.tensor(test_accs)
    print('============================')
    print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')

w.tick('finish')
del w
rpc.shutdown()
