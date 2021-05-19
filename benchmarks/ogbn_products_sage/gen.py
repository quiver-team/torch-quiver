import torch

import argparse


class ProductsDataset:
    def __init__(self, train_idx, edge_index, x, y, f, c):
        self.train_idx = train_idx
        self.edge_index = edge_index
        self.x = x
        self.y = y
        self.num_features = f
        self.num_classes = c


p = argparse.ArgumentParser(description='')
p.add_argument('--nodes', type=int, default=2000000, help='number of nodes')
p.add_argument('--train',
               type=int,
               default=200000,
               help='number of train nodes')
p.add_argument('--edges', type=int, default=25, help='number of edges')
p.add_argument('--features', type=int, default=100, help='number of features')
p.add_argument('--labels', type=int, default=100, help='number of labels')
args = p.parse_args()

train_dst = torch.randint(args.train, (args.train, ))
train_src = torch.arange(args.train, dtype=torch.int64)

train_edges = torch.stack([train_src, train_dst])

other_edges = torch.randint(args.nodes, (2, args.nodes * args.edges * 2))

total_edges = torch.cat([train_edges, other_edges], dim=1)

node_features = torch.randint(1000000, (args.nodes, args.features),
                              dtype=torch.float)

node_labels = torch.randint(args.labels, (args.nodes, ))

torch.save(
    ProductsDataset(train_src, total_edges, node_features, node_labels,
                    args.features, args.labels), 'products.pt')
