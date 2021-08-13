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
import time


from typing import List, NamedTuple, Optional, Tuple


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


class SampleRequest:
    def __init__(self, src, dst, nodes, size):
        self.index = index
        self.src = src
        self.dst = dst
        self.nodes = nodes
        self.size = size


class SampleResponse:
    def __init__(self, src, dst, outputs, counts):
        self.src = src
        self.dst = dst
        self.outputs = outputs
        self.counts = counts


class FeatureRequest:
    def __init__(self, src, dst, nodes):
        self.src = src
        self.dst = dst
        self.nodes = nodes


class FeatureResponse:
    def __init__(self, src, dst, features):
        self.src = src
        self.dst = dst
        self.features = features


class SingleProcess:
    def __init__(self, num_epoch, num_batch, *args):
        self.num_epoch = num_epoch
        self.num_batch = num_batch
        self.args = args

    def prepare(self, rank, sample_data, train_data, feature_data, sync, comm):
        device, edge_index, train_idx, sizes, batch_size = sample_data
        torch.cuda.set_device(device)
        self.sync = sync
        self.comm = comm
        self.comm.rank = rank
        self.train_idx = train_idx
        self.batch_size = batch_size
        self.loader = AsyncCudaNeighborSampler(edge_index,
                                               device=device,
                                               node_idx=train_idx,
                                               sizes=sizes,
                                               batch_size=batch_size,
                                               shuffle=True)
        self.sizes = sizes
        num_features, num_hidden, num_classes, num_layers, y = train_data
        self.y = y
        device = torch.device('cuda:' +
                              str(self.comm.rank) if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.feature = feature_data.to(device)
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

    def dispatch(self, nodes, ws):
        ranks = torch.fmod(nodes, ws)
        input_orders = torch.arange(nodes.size(
            0), dtype=torch.long, device=nodes.device)
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
            total_inputs = []
            sample_reorder, sample_args = self.dispatch(nodes, self.comm.ws)
            sample_results = []
            for rank, part_nodes in enumerate(sample_args):
                if rank == self.comm.rank:
                    result = self.loader.sample_layer(part_nodes, size)
                else:
                    result = None
                    req = SampleRequest(self.comm.rank, rank, part_nodes, size)
                    self.sync.request_queues[rank].put(req)
                sample_results.append(result)
                total_inputs.append(part_nodes)
            total_inputs = torch.cat(total_inputs)
            #self.sync.peer_barrier.wait()
            for i in range(self.comm.ws - 1):
                req = self.sync.request_queues[self.comm.rank].get()
                src = req.src
                dst = req.dst
                part_nodes = req.nodes.to(self.device)
                # to local
                out, cnt = self.loader.sample_layer(part_nodes, size)
                # to global
                resp = SampleResponse(src, dst, out, cnt)
                self.sync.response_queue[src].put(resp)
            for i in range(self.comm.ws - 1):
                resp = self.sync.response_queues[self.comm.rank].get()
                src = resp.src
                dst = resp.dst
                out = resp.outputs
                cnt = resp.counts
                sample_results[dst] = out.to(self.device), cnt.to(self.device)
            total_outputs = []
            total_counts = []
            for out, cnt in sample_results:
                total_outputs.append(out)
                total_counts.append(cnt)
            total_outputs = torch.cat(total_outputs)
            total_counts = torch.cat(total_counts)
            frontier, row_idx, col_idx = self.loader.reindex(sample_reorder, total_inputs, total_outputs, total_counts)
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
        feature_reorder, feature_args = self.dispatch(nodes, self.comm.ws)
        feature_results = []
        for rank, part_nodes in enumerate(feature_args):
            if rank == self.comm.rank:
                result = self.features[part_nodes]
            else:
                result = None
                req = FeatureRequest(self.comm.rank, rank, part_nodes)
                self.sync.request_queues[rank].put(req)
            feature_results.append(result)
            total_inputs.append(part_nodes)
        for i in range(self.comm.ws - 1):
            req = self.sync.request_queues[self.comm.rank].get()
            src = req.src
            dst = req.dst
            part_nodes = req.nodes.to(self.device)
            # to local
            feature = self.feature[part_nodes]
            # to global
            resp = FeatuerResponse(src, dst, feature)
            self.sync.response_queue[src].put(resp)
        for i in range(self.comm.ws - 1):
            resp = self.sync.response_queues[self.comm.rank].get()
            src = resp.src
            dst = resp.dst
            feature = resp.feature
            feature_results[dst] = feature.to(self.device)
        total_features = []
        for feature in feature_results:
            total_features.append(feature)
        total_features = torch.cat(total_features)
        total_features[feature_reorder] = total_features
        return total_features


    def __call__(self, rank):
        self.prepare(rank, *self.args)
        dataloader = torch.utils.data.DataLoader(
            self.train_idx, batch_size=self.batch_size, shuffle=True, drop_last=True)
        for i in range(epoch):
            count = 0
            cont = True
            while cont:
                for data in dataloader:
                    count += 1
                    n_id, batch_size, adjs = self.sample(data)
                    features = self.collect(nodes)
                    self.optimizer.zero_grad()
                    out = self.model(features, adjs)
                    label_ids = n_id[:batch_size].to(self.device)
                    loss = F.nll_loss(out, self.y[label_ids].to(self.device))
                    loss.backward()
                    self.optimizer.step()
                    if count >= num_batch:
                        cont = False
                        break

if __name__ == '__main__':
    mp.set_start_method('spawn')
    proc = SingleProcess()
    procs = launch_multiprocess(proc, 4)
    