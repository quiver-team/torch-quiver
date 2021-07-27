#!/usr/bin/env python3
# modified from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_products_sage.py
# Reaches around 0.7870 ± 0.0036 test accuracy.

import argparse
import multiprocessing as mp
import os
import os.path as osp

import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from quiver.cuda_sampler import CudaNeighborSampler
from quiver.cuda_loader import CudaNeighborLoader
from quiver.profile_utils import StopWatch
from quiver.models.sage_model import SAGE


class ProductsDataset:
    def __init__(self, train_idx, edge_index, x, y, f, c):
        self.train_idx = train_idx
        self.edge_index = edge_index
        self.x = x
        self.y = y
        self.num_features = f
        self.num_classes = c


def main():
    p = argparse.ArgumentParser(description='')
    p.add_argument('--mode',
                   type=str,
                   default='sync',
                   help='sync | await | coro | prefetch')
    p.add_argument('--runs', type=int, default=10, help='number of runs')
    p.add_argument('--epochs', type=int, default=20, help='number of epochs')
    p.add_argument('--distribute',
                   type=str,
                   default='',
                   help='kungfu | horovod')
    args = p.parse_args()

    if args.distribute == 'horovod':
        import horovod.torch as hvd
        hvd.init()
        if torch.cuda.is_available():
            torch.cuda.set_device(hvd.local_rank())

    print('loading ... ')
    w = StopWatch('main')
    root = './products.pt'
    dataset = torch.load(root)
    evaluator = Evaluator(name='ogbn-products')

    train_idx = dataset.train_idx

    w.tick('load data')
    train_loader = None

    if args.mode == 'prefetch':
        train_loader = CudaNeighborLoader(
            (dataset.edge_index, [15, 10, 5], train_idx), 1024, 4)
    else:
        train_loader = CudaNeighborSampler(dataset.edge_index,
                                           node_idx=train_idx,
                                           sizes=[15, 10, 5],
                                           batch_size=1024,
                                           mode=args.mode,
                                           shuffle=True)
    w.tick('create train_loader')
    subgraph_loader = CudaNeighborSampler(dataset.edge_index,
                                          node_idx=None,
                                          sizes=[-1],
                                          batch_size=4096,
                                          shuffle=False)
    w.tick('create subgraph_loader')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=3)
    model = model.to(device)

    x = dataset.x.to(device)  # [N, 100]
    y = dataset.y.squeeze().to(device)  # [N, 1]
    w.tick('build model')

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
        w.tick('before for loop')
        for epoch in range(1, 1 + args.epochs):
            #loss, acc = train(epoch)
            loss, acc = model.train_m(train_loader, w, optimizer, device, x, y,
                                      train_idx, epoch, args.mode, args.epochs)
            print(
                f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}'
            )
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


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
