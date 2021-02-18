#!/usr/bin/env python3
# modified from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_products_sage.py
# Reaches around 0.7870 ± 0.0036 test accuracy.

import argparse
import os
import os.path as osp

import torch
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from quiver.profile_utils import StopWatch
from torch_geometric.data import NeighborSampler
from quiver.models.sage_model import SAGE

p = argparse.ArgumentParser(description='')
p.add_argument('--num-workers',
               type=int,
               default=1,
               help='number of CPU workers')
p.add_argument('--runs', type=int, default=10, help='number of runs')
p.add_argument('--epochs', type=int, default=20, help='number of epochs')
p.add_argument('--distribute', type=str, default='', help='kungfu | horovod')
args = p.parse_args()

if args.distribute == 'horovod':
    import horovod.torch as hvd
    hvd.init()
    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())

print('loading ... ')
w = StopWatch('main')
home = os.getenv('HOME')
data_dir = osp.join(home, '.pyg')
root = osp.join(data_dir, 'data', 'products')
dataset = PygNodePropPredDataset('ogbn-products', root)
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-products')
data = dataset[0]

train_idx = split_idx['train']

w.tick('load data')
train_loader = NeighborSampler(data.edge_index,
                               node_idx=train_idx,
                               sizes=[15, 10, 5],
                               batch_size=1024,
                               shuffle=True,
                               num_workers=args.num_workers)
w.tick('create train_loader')
subgraph_loader = NeighborSampler(data.edge_index,
                                  node_idx=None,
                                  sizes=[-1],
                                  batch_size=4096,
                                  shuffle=False,
                                  num_workers=args.num_workers)

w.tick('create subgraph_loader')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=3)
model = model.to(device)

x = data.x.to(device)  # [N, 100]
y = data.y.squeeze().to(device)  # [N, 1]
w.tick('build model')

# def train():
#     # w1 = StopWatch('train loop')
#     model.train()
#     # w1.tick('set mode to train')
#
#     # pbar = tqdm(total=train_idx.size(0))
#     # pbar.set_description(f'Epoch {epoch:02d}')
#
#     total_loss = total_correct = 0
#     w.turn_on('sample')
#     for batch_size, n_id, adjs in train_loader:
#         w.turn_off('sample')
#         # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
#         # w1.tick('prepro')
#         adjs = [adj.to(device) for adj in adjs]
#
#         optimizer.zero_grad()
#         out = model(x[n_id], adjs)
#         loss = F.nll_loss(out, y[n_id[:batch_size]])
#         loss.backward()
#         optimizer.step()
#         # w1.tick('train')
#
#         total_loss += float(loss)
#         total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
#         # pbar.update(batch_size)
#         w.turn_on('sample')
#
#     # pbar.close()
#
#     loss = total_loss / len(train_loader)
#     approx_acc = total_correct / train_idx.size(0)
#
#     # del w1
#     return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x, subgraph_loader, device)

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


test_accs = []
for run in range(1, 1 + args.runs):
    print('')
    print(f'Run {run:02d}:')
    print('')

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    if args.distribute == 'kungfu':
        import kungfu.torch as kf
        optimizer = kf.optimizers.SynchronousSGDOptimizer(
            optimizer, named_parameters=model.named_parameters())
        # Broadcast parameters from rank 0 to all other processes.
        kf.broadcast_parameters(model.state_dict())
    elif args.distribute == 'horovod':
        optimizer = hvd.DistributedOptimizer(optimizer)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    best_val_acc = final_test_acc = 0.0
    w.tick('?')
    for epoch in range(1, 1 + args.epochs):
        loss, acc = model.train_m(train_loader, w, optimizer, device, x, y,
                                  train_idx, epoch, "sync", args.epochs)
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
