import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from quiver.async_cuda_sampler import AsyncCudaNeighborSampler
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

import kungfu.torch as kf
from kungfu.python import current_cluster_size, current_rank
from kungfu.cmd import launch_multiprocess

import os
import os.path as osp
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from quiver.models.sage_model import SAGE
import time

from typing import List, NamedTuple, Optional, Tuple

NUMA_SIZE = 2
PER_NUMA = 2


class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size)


class CommConfig:
    def __init__(self, rank, ws):
        self.rank = rank
        self.ws = ws


class SyncManager:
    def __init__(self, ws):
        self.request_queues = [mp.Queue(64) for i in range(ws)]
        self.response_queues = [mp.Queue(64) for i in range(ws)]
        self.peer_barrier = mp.Barrier(ws)


class FeatureResponse:
    def __init__(self, src, rank, feature):
        self.src = src
        self.rank = rank
        self.feature = feature


class SingleProcess:
    def __init__(self, num_epoch, num_batch, *args):
        self.num_epoch = num_epoch
        self.num_batch = num_batch
        self.args = args

    def prepare(self, rank, sample_data, train_data, feature_data, sync, comm):
        edge_index, batch_size, sizes, train_idx = sample_data
        device = rank
        torch.cuda.set_device(device)
        self.sync = sync
        self.comm = comm
        self.comm.rank = rank
        self.train_idx = train_idx
        self.batch_size = batch_size
        self.loader = AsyncCudaNeighborSampler(edge_index, device=device)
        self.sizes = sizes
        num_features, num_hidden, num_classes, num_layers, y = train_data
        self.y = y
        device = torch.device(
            'cuda:' +
            str(self.comm.rank) if torch.cuda.is_available() else 'cpu')
        self.device = device
        feature_rank = rank % PER_NUMA + 1
        numa_rank = rank // PER_NUMA
        feature_peers = list(
            range(numa_rank * PER_NUMA, (numa_rank + 1) * PER_NUMA))
        features = feature_data
        cpu_features = features[0]
        local_feature = features[feature_rank]
        self.feature_rank = feature_rank
        self.feature_peers = feature_peers
        self.feature = local_feature.to(device)
        model = SAGE(num_features, num_hidden, num_classes, num_layers)
        model = model.to(device)
        model.reset_parameters()

        model.train()
        self.model = model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        optimizer = kf.optimizers.SynchronousSGDOptimizer(
            optimizer, named_parameters=model.named_parameters())
        kf.broadcast_parameters(model.state_dict())
        self.optimizer = optimizer
        for peer in feature_peers:
            if peer != self.comm.rank:
                resp = FeatureResponse(self.comm.rank, self.feature_rank,
                                       self.feature)
                self.sync.response_queues[peer].put(resp)
        all_features = [None] * (len(feature_peers) + 1)
        all_features[0] = cpu_feature
        all_features[self.comm.rank] = self.feature
        for i in range(len(feature_peers)):
            resp = self.sync.response_queues[self.comm.rank].get()
            all_features[resp.rank] = resp.feature
        # TODO: init shard features
        self.shard_features = f(all_features)

    def dispatch(self, nodes, ws):
        ranks = torch.fmod(nodes, ws)
        input_orders = torch.arange(nodes.size(0),
                                    dtype=torch.long,
                                    device=nodes.device)
        reorder = torch.empty_like(input_orders)
        beg = 0
        res = []
        for i in range(ws):
            mask = torch.eq(ranks, i)
            part_nodes = torch.masked_select(nodes, mask)
            part_orders = torch.masked_select(input_orders, mask)
            reorder[beg:beg + part_nodes.size(0)] = part_orders
            beg += part_nodes.size(0)
            res.append(part_nodes)
        return reorder, res

    def sample(self, nodes):
        batch_size = len(nodes)
        nodes = nodes.to(self.device)
        adjs = []
        for size in self.sizes:
            out, cnt = self.loader.sample_layer(nodes, size)
            frontier, row_idx, col_idx = self.loader.reindex(nodes, out, cnt)
            row_idx, col_idx = col_idx, row_idx
            edge_index = torch.stack([row_idx, col_idx], dim=0)

            adj_size = torch.LongTensor([
                frontier.size(0),
                nodes.size(0),
            ])
            e_id = torch.tensor([])
            adjs.append(Adj(edge_index, e_id, adj_size))
            nodes = frontier
        return nodes, batch_size, adjs[::-1]

    def collect(self, nodes):
        nodes = nodes.to(self.device)
        total_features = self.shard_features[nodes]
        return total_features

    def __call__(self, rank):
        self.prepare(rank, *self.args)
        dataloader = torch.utils.data.DataLoader(self.train_idx,
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 drop_last=True)
        for i in range(self.num_epoch):
            count = 0
            cont = True
            while cont:
                nodes_list = []
                t0 = time.time()
                for data in dataloader:
                    n_id, batch_size, adjs = self.sample(data)
                    t1 = time.time()
                    features = self.collect(n_id)
                    t2 = time.time()
                    self.optimizer.zero_grad()
                    out = self.model(features, adjs)
                    label_ids = n_id[:batch_size].to(self.device)
                    loss = F.nll_loss(out, self.y[label_ids].to(self.device))
                    loss.backward()
                    self.optimizer.step()
                    # print(f'rank {self.comm.rank} sample {t1 - t0}')
                    # print(f'rank {self.comm.rank} feature {t2 - t1}')
                    # print(f'rank {self.comm.rank} took {time.time() - t0}')
                    t0 = time.time()
                    count += 1
                    if count >= self.num_batch:
                        cont = False
                        break


if __name__ == '__main__':
    mp.set_start_method('spawn')
    ws = PER_NUMA * NUMA_SIZE
    num_epoch = 1
    num_batch = 200
    batch_size = 128
    sizes = [15, 10, 5]
    home = os.getenv('HOME')
    data_dir = osp.join(home, '.pyg')
    root = osp.join(data_dir, 'data', 'products')
    # root = "/home/dalong/data"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    train_idx = split_idx['train']
    edge_index = data.edge_index
    x, y = data.x, data.y.squeeze().share_memory_()
    sample_data = edge_index, batch_size, sizes, train_idx
    train_data = dataset.num_features, 256, dataset.num_classes, 3, y
    comm = CommConfig(0, ws)
    sync = SyncManager(ws)
    per_numa_section = [1, 2, 2]
    per_numa_features = torch.split(x, per_numa_section, dim=0)
    proc = SingleProcess(num_epoch, num_batch, sample_data, train_data,
                         per_numa_features, sync, comm)
    procs = launch_multiprocess(proc, ws)
    time.sleep(50)
    for p in procs:
        p.kill()
