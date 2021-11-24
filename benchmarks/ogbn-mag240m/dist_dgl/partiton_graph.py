import dgl
import numpy as np
import torch as th
import argparse
import time

from scipy.sparse import csc_matrix
import torch
import quiver
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from ogb.lsc import MAG240MDataset
import os.path as osp
from torch_geometric.datasets import Reddit
from dgl.data.tensor_serialize import save_tensors, load_tensors


def load_mag240M():
    indptr = torch.load("/data/mag/mag240m_kddcup2021/csr/indptr.pt")
    indices = torch.load("/data/mag/mag240m_kddcup2021/csr/indices.pt")
    dataset = MAG240MDataset("/data/mag")
    train_idx = torch.from_numpy(dataset.get_idx_split('train'))
    val_idx = torch.from_numpy(dataset.get_idx_split('validate'))
    test_idx = torch.from_numpy(dataset.get_idx_split('test'))

    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)

    train_mask = torch.zeros(csr_topo.indptr.shape[0]-1, dtype=torch.uint8)
    train_mask[train_idx] = 1
    return train_mask, csr_topo

def load_paper100M():
    indptr = torch.load("/data/papers/ogbn_papers100M/csr/indptr_bi.pt")
    indices = torch.load("/data/papers/ogbn_papers100M/csr/indices_bi.pt")
    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    train_mask = torch.ones(csr_topo.indptr.shape[0]-1, dtype=torch.uint8)
    
    return train_mask, csr_topo

def load_products():
    root = "/home/dalong/data/products/"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    train_mask = torch.ones(csr_topo.indptr.shape[0]-1, dtype=torch.uint8)
    train_mask[train_idx] = 1

    val_idx = split_idx['valid']
    val_mask = torch.ones(csr_topo.indptr.shape[0]-1, dtype=torch.uint8)
    val_mask[val_idx] = 1

    test_idx = split_idx['test']
    test_mask = torch.ones(csr_topo.indptr.shape[0]-1, dtype=torch.uint8)
    test_mask[test_idx] = 1


    label = data.y.squeeze()
    feature = data.x

    return (train_mask, val_mask, test_mask), csr_topo, feature, label

def load_reddit():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    dataset = Reddit(path)
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)
    feature = data.x
    label = data.y

    return (data.train_mask, data.val_mask, data.test_mask), csr_topo, feature, label


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument('--dataset', type=str, default='reddit',
                           help='datasets: reddit, ogb-product, ogb-paper100M')
    argparser.add_argument('--num_parts', type=int, default=4,
                           help='number of partitions')
    argparser.add_argument('--part_method', type=str, default='random',
                           help='the partition method')
    argparser.add_argument('--balance_train', action='store_true',
                           help='balance the training size in each partition.')
    argparser.add_argument('--undirected', action='store_true',
                           help='turn the graph into an undirected graph.')
    argparser.add_argument('--balance_edges', action='store_true',
                           help='balance the number of edges in each partition.')
    argparser.add_argument('--num_trainers_per_machine', type=int, default=1,
                           help='the number of trainers per machine. The trainer ids are stored\
                                in the node feature \'trainer_id\'')
    argparser.add_argument('--output', type=str, default='data',
                           help='Output path of partitioned graph.')
    args = argparser.parse_args()

    start = time.time()
    if args.dataset == 'reddit':
        data_split, csr_topo, feature, label = load_reddit()
    elif args.dataset == 'ogb-product':
        data_split, csr_topo, feature, label = load_products()
    elif args.dataset == 'ogb-paper100M':
        data_split, csr_topo, feature, label = load_paper100M()
    elif args.dataset == 'ogb-mag240M':
        data_split, csr_topo, feature, label = load_mag240M()
    print('load {} takes {:.3f} seconds'.format(args.dataset, time.time() - start))
    
    index = np.zeros(csr_topo.indices.size(0), dtype=np.int8)
    nodes = csr_topo.indptr.size(0) - 1
    csc = csc_matrix((index, csr_topo.indices.numpy(), csr_topo.indptr.numpy()),shape=[nodes, nodes])
    train_g = dgl.from_scipy(csc)

    train_g.ndata['train_mask'] = data_split[0]
    train_g.ndata['val_mask'] = data_split[1]
    train_g.ndata['test_mask'] = data_split[2]
    train_g.ndata["labels"] = label
    train_g.ndata["features"] = feature

    if args.balance_train:
        balance_ntypes = train_g.ndata['train_mask']
    else:
        balance_ntypes = None
        
    dgl.distributed.partition_graph(train_g, args.dataset, args.num_parts, args.output,
                                    part_method=args.part_method,
                                    balance_ntypes=balance_ntypes,
                                    balance_edges=args.balance_edges,
                                    num_trainers_per_machine=args.num_trainers_per_machine)