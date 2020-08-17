#!/usr/bin/env python3

import os

import torch


def info(t, name=None):
    msg = ''
    if name:
        msg += name
    msg += ' ' + str(t.type())
    msg += ' ' + str(t.shape)
    print(msg)


def main():
    home = os.getenv('HOME')
    data_dir = os.path.join(home, '.pyg')
    root = os.path.join(data_dir, 'data', 'products')
    filename = os.path.join(
        root, 'ogbn_products_pyg/processed/geometric_data_processed.pt')
    print(filename)
    data, kvs = torch.load(filename)
    print(data.__class__)  # torch_geometric.data.data.Data

    for idx, d in enumerate(dir(data)):
        print('[%d]=%s' % (idx, d))

    info(data.x, 'data.x')
    info(data.y, 'data.y')
    info(data.edge_index, 'data.edge_index')

    for k, v in kvs.items():
        print(k)
        print(v.__class__)

    info(kvs['x'], 'x')
    info(kvs['y'], 'y')
    info(kvs['edge_index'], 'edge_index')

    print(kvs['x'])  # tensor([      0, 2449029])
    print(kvs['y'])  # tensor([      0, 2449029])
    print(kvs['edge_index'])  # tensor([        0, 123718280])


main()
