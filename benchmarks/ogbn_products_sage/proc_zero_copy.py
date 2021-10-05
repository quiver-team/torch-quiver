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
import numpy as np
from scipy.sparse import csr_matrix

from typing import List, NamedTuple, Optional, Tuple
from numa import schedule, info


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


class BatchSampleRequest:
    def __init__(self, index, src, dst, nodes, size):
        self.index = index
        self.src = src
        self.dst = dst
        self.nodes = nodes
        self.size = size


class SampleResponse:
    def __init__(self, index, src, dst, outputs, counts):
        self.index = index
        self.src = src
        self.dst = dst
        self.outputs = outputs
        self.counts = counts


class ReindexRequest:
    def __init__(self, index, src, dst, reorder, inputs, outputs, counts):
        self.index = index
        self.src = src
        self.dst = dst
        self.reorder = reorder
        self.inputs = inputs
        self.outputs = outputs
        self.counts = counts


class ReindexResponse:
    def __init__(self, index, src, dst, nodes, row, col):
        self.index = index
        self.src = src
        self.dst = dst
        self.nodes = nodes
        self.row = row
        self.col = col


class FeatureRequest:
    def __init__(self, index, src, dst, nodes):
        self.index = index
        self.src = src
        self.dst = dst
        self.nodes = nodes


class FeatureResponse:
    def __init__(self, index, src, dst, features):
        self.index = index
        self.src = src
        self.dst = dst
        self.features = features


class SingleProcess:
    def __init__(self, num_epoch, num_batch, *args):
        self.num_epoch = num_epoch
        self.num_batch = num_batch
        self.args = args

    def prepare(self, rank, sample_data, train_data, feature_data, sync, comm):
        #####################################################################################################
        # Bind Task To NUMA Node And Sleep 1s So That Next Time This Processing Is Runing On Target NUMA Node
        #####################################################################################################
        total_nodes = info.get_max_node() + 1
        schedule.bind(rank % total_nodes)
        print(
            f"LOG >>> Rank {rank} Is Bind To NUMA Node {rank % total_nodes}/{total_nodes}"
        )
        time.sleep(1)

        csr_mat, batch_size, sizes, train_idx = sample_data
        device = rank
        torch.cuda.set_device(device)
        self.list_size = 10
        self.sample_count = 0
        self.handle_time = 0
        self.sample_time = 0
        self.reindex_time = 0
        self.reindex_count = 0
        self.queue_time = [0, 0, 0, 0]
        self.queue_count = 0
        self.request_time = 0
        self.response_time = 0
        self.recv_time = 0
        self.ready = False
        self.sync = sync
        self.comm = comm
        self.comm.rank = rank
        self.train_idx = train_idx
        self.batch_size = batch_size
        self.indptr = torch.from_numpy(csr_mat.indptr[:-1]).type(torch.long)
        self.indices = torch.from_numpy(csr_mat.indices).type(torch.long)
        self.loader = AsyncCudaNeighborSampler(csr_indptr=self.indptr,
                                               csr_indices=self.indices,
                                               device=device,
                                               copy=True)
        self.sizes = sizes

    def run(self, *args):
        pass

    def handle_sample_response(self, sample_response):
        batch_id = sample_response.index.batch_id
        rank = sample_response.index.rank
        size = sample_response.index.size
        outputs = sample_response.outputs
        counts = sample_response.counts
        ret = self.manager.recv_sample(batch_id, rank, size, outputs, counts)
        if ret is not None:
            reorder, inputs, outputs, counts = ret
            index = MicroBatchIndex(batch_id, rank, size)
            reindex_peer = self.comm.other_peers.local_reindex
            request = ReindexRequest(index, self.global_rank, reindex_peer,
                                     reorder, inputs, outputs, counts)
            self.sync.upstream_queues[reindex_peer].put(request)

    def handle_reindex_response(self, reindex_response):
        batch_id = reindex_response.index.batch_id
        nodes = reindex_response.nodes
        row = reindex_response.row
        col = reindex_response.col
        ret = self.manager.recv_reindex(batch_id, nodes, row, col)
        nodes, temp_layer = ret
        if temp_layer >= 0:
            reorder, res = self.manager.dispatch(nodes, self.sample_size,
                                                 False)
            res = self.manager.prepare_request(batch_id, reorder, res, False)
            global_samplers = self.comm.other_peers.global_samplers
            for rank, nodes in enumerate(res):
                index = MicroBatchIndex(batch_id, rank, len(res))
                req = SamplerRequest(index, self.comm.global_rank,
                                     global_samplers[rank], nodes,
                                     self.sizes[temp_layer])
                self.sync.request_queues[global_samplers[rank]].put(req)
        else:
            reorder, res = self.manager.dispatch(nodes, self.feature_size,
                                                 True)
            res = self.manager.prepare_request(batch_id, reorder, res, True)
            group_peers = self.comm.group_peers
            for rank, nodes in enumerate(res):
                index = MicroBatchIndex(batch_id, rank, len(res))
                req = FeatureRequest(index, self.comm.global_rank,
                                     group_peers[rank], nodes)
                self.sync.request_queues[group_peers[rank]].put(req)

    def handle_feature_request(self, feature_request):
        index = feature_request.index
        nodes = feature_request.nodes
        src = feature_request.src
        dst = feature_request.dst
        features = self.manager.features[nodes]
        response = FeatureResponse(index, src, dst, features)
        self.sync.response_queues[dst].put(response)

    def handle_feature_response(self, feature_respnse):
        batch_id = feature_request.index.batch_id
        features = feature_request.features
        rank = feature_request.index.rank
        size = feature_request.index.size
        features = self.manager.recv_feature(batch_id, rank, size, features)
        if features:
            batch_size, n_ids, adjs = self.manager.prepare_train()
            args = TrainArgs(batch_size, n_ids, adjs, features)
            trainer = self.comm.downstream_peer
            self.sync.upstream_queues[trainer].put(args)
            del self.manager.buffers[batch_id]


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
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)


class TrainerProcess(QuiverProcess):
    def prepare(self, data):
        num_features, num_hidden, num_classes, num_layers, y = data
        self.y = y
        device = torch.device(
            'cuda:' +
            str(self.comm.group_rank) if torch.cuda.is_available() else 'cpu')
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

    def batch_sample(self, nodes_list):
        batch_size_list = [len(nodes) for nodes in nodes_list]
        nodes_list = [nodes.to(self.device) for nodes in nodes_list]
        adjs_list = [[] for nodes in nodes_list]
        for size in self.sizes:
            total_inputs_list = []
            sample_reorder_list = []
            sample_input_list = [None] * (self.comm.ws * self.list_size)
            sample_results_list = [[None] * self.comm.ws
                                   for i in range(self.list_size)]
            t0 = time.time()
            for j in range(self.list_size):
                nodes = nodes_list[j]
                total_inputs = []
                sample_reorder, sample_args = self.dispatch(
                    nodes, self.comm.ws)
                sample_reorder_list.append(sample_reorder)
                sample_results = []
                for rank, part_nodes in enumerate(sample_args):
                    if rank == self.comm.rank:
                        sample_input_list[self.comm.rank * self.list_size +
                                          j] = part_nodes
                    else:
                        req = BatchSampleRequest(j, self.comm.rank, rank,
                                                 part_nodes, size)
                        self.sync.request_queues[rank].put(req)
                    total_inputs.append(part_nodes)
                total_inputs = torch.cat(total_inputs)
                total_inputs_list.append(total_inputs)
            t1 = time.time()
            for j in range(self.list_size):
                for i in range(self.comm.ws - 1):
                    q_beg = time.time()
                    req = self.sync.request_queues[self.comm.rank].get()
                    self.queue_count += 1
                    index = req.index
                    src = req.src
                    dst = req.dst
                    if self.ready:
                        self.queue_time[src] += time.time() - q_beg
                    part_nodes = req.nodes.to(self.device)
                    sample_input_list[src * self.list_size +
                                      index] = part_nodes
                    # s_beg = time.time()
                    # out, cnt = self.loader.sample_layer(part_nodes, size)
                    # self.sample_time += time.time() - s_beg
                    # self.sample_count += 1
                    # resp = SampleResponse(src, dst, out, cnt)
                    # self.sync.response_queues[src].put(resp)
            t2 = time.time()
            batch_sample_input = torch.cat(sample_input_list)
            batch_out, batch_cnt = self.loader.sample_layer(
                batch_sample_input, size)
            t3 = time.time()
            batch_prefix = torch.cumsum(batch_cnt, dim=0)
            # [1,2,3,4,5,6,7,8]
            # [2,2,2,2]
            # [2,4,6,8]
            size_beg = 0
            for i in range(self.comm.ws):
                for j in range(self.list_size):
                    single_size = len(sample_input_list[i * self.list_size +
                                                        j])  #1
                    single_cnt = batch_cnt[size_beg:size_beg +
                                           single_size]  #[2]
                    if i == 0 and j == 0:
                        prefix_beg = 0
                    else:
                        prefix_beg = batch_prefix[size_beg - 1]  #0 2
                    prefix_end = batch_prefix[size_beg + single_size - 1]  #2 4
                    single_out = batch_out[prefix_beg:prefix_end]  # [0:2]
                    size_beg += single_size
                    if i == self.comm.rank:
                        sample_results_list[j][i] = (single_out, single_cnt)
                    else:
                        resp = BatchSampleResponse(j, i, self.comm.rank,
                                                   single_out, single_cnt)
                        self.sync.response_queues[i].put(resp)
            for i in range(self.comm.ws - 1):
                for j in range(self.list_size):
                    q_beg = time.time()
                    resp = self.sync.response_queues[self.comm.rank].get()
                    index = resp.index
                    src = resp.src
                    dst = resp.dst
                    out = resp.outputs
                    cnt = resp.counts
                    if self.ready:
                        self.queue_time[src] += time.time() - q_beg
                    sample_results_list[index][dst] = out.to(
                        self.device), cnt.to(self.device)
            t4 = time.time()
            for j in range(self.list_size):
                total_outputs = []
                total_counts = []
                for out, cnt in sample_results_list[j]:
                    total_outputs.append(out)
                    total_counts.append(cnt)
                total_outputs = torch.cat(total_outputs)
                total_counts = torch.cat(total_counts)
                frontier, row_idx, col_idx = self.loader.reindex(
                    sample_reorder_list[j], total_inputs_list[j],
                    total_outputs, total_counts)
                self.reindex_count += 1
                row_idx, col_idx = col_idx, row_idx
                edge_index = torch.stack([row_idx, col_idx], dim=0)

                adj_size = torch.LongTensor([
                    frontier.size(0),
                    nodes_list[j].size(0),
                ])
                e_id = torch.tensor([])
                adjs_list[j].append(Adj(edge_index, e_id, adj_size))
                nodes_list[j] = frontier
            t5 = time.time()
            if self.ready:
                self.request_time += t1 - t0
                self.sample_time += t3 - t2
                self.recv_time += t2 - t1
                self.handle_time += t4 - t3
                self.reindex_time += t5 - t4
            self.ready = True
        adjs_list = [adjs[::-1] for adjs in adjs_list]
        return nodes_list, batch_size_list, adjs_list

    def sample(self, input_nodes):
        nodes = input_nodes.to(self.device)
        adjs = []

        batch_size = len(nodes)
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
            # t0 = time.time()
            # total_inputs = []
            # sample_reorder, sample_args = self.dispatch(nodes, self.comm.ws)
            # sample_results = []
            # for rank, part_nodes in enumerate(sample_args):
            #     if rank == self.comm.rank:
            #         s_beg = time.time()
            #         result = self.loader.sample_layer(part_nodes, size)
            #         self.sample_count += 1
            #         if self.sample_count > 1:
            #             self.sample_time += time.time() - s_beg
            #     else:
            #         result = None
            #         req = SampleRequest(self.comm.rank, rank, part_nodes, size)
            #         self.sync.request_queues[rank].put(req)
            #     sample_results.append(result)
            #     total_inputs.append(part_nodes)
            # total_inputs = torch.cat(total_inputs)
            # t1 = time.time()
            # for i in range(self.comm.ws - 1):
            #     q_beg = time.time()
            #     req = self.sync.request_queues[self.comm.rank].get()
            #     self.queue_count += 1
            #     src = req.src
            #     dst = req.dst
            #     if self.queue_count > 4:
            #         self.queue_time[src] += time.time() - q_beg
            #     part_nodes = req.nodes.to(self.device)
            #     # to local
            #     s_beg = time.time()
            #     out, cnt = self.loader.sample_layer(part_nodes, size)
            #     self.sample_time += time.time() - s_beg
            #     self.sample_count += 1
            #     # to global
            #     resp = SampleResponse(src, dst, out, cnt)
            #     self.sync.response_queues[src].put(resp)
            # t2 = time.time()
            # for i in range(self.comm.ws - 1):
            #     resp = self.sync.response_queues[self.comm.rank].get()
            #     src = resp.src
            #     dst = resp.dst
            #     out = resp.outputs
            #     cnt = resp.counts
            #     sample_results[dst] = out.to(self.device), cnt.to(self.device)
            # total_outputs = []
            # total_counts = []
            # t3 = time.time()
            # for out, cnt in sample_results:
            #     total_outputs.append(out)
            #     total_counts.append(cnt)
            # total_outputs = torch.cat(total_outputs)
            # total_counts = torch.cat(total_counts)
            # r_beg = time.time()
            # frontier, row_idx, col_idx = self.loader.reindex(
            #     sample_reorder, total_inputs, total_outputs, total_counts)
            # self.reindex_count += 1
            # if self.reindex_count > 1:
            #     self.reindex_time += time.time() - r_beg
            # row_idx, col_idx = col_idx, row_idx
            # edge_index = torch.stack([row_idx, col_idx], dim=0)

            # adj_size = torch.LongTensor([
            #     frontier.size(0),
            #     nodes.size(0),
            # ])
            # e_id = torch.tensor([])
            # adjs.append(Adj(edge_index, e_id, adj_size))
            # nodes = frontier
            # if self.reindex_count > 1:
            #     self.request_time += t1 - t0
            #     self.response_time += t2 - t1
            #     self.recv_time += t3 - t2
        return nodes, batch_size, adjs[::-1]

    def collect(self, nodes):
        nodes = nodes.to(self.device)
        feature_reorder, feature_args = self.dispatch(nodes, self.comm.ws)
        feature_results = []
        for rank, part_nodes in enumerate(feature_args):
            if rank == self.comm.rank:
                result = self.feature[part_nodes]
            else:
                result = None
                req = FeatureRequest(self.comm.rank, rank, part_nodes)
                self.sync.request_queues[rank].put(req)
            feature_results.append(result)
        for i in range(self.comm.ws - 1):
            req = self.sync.request_queues[self.comm.rank].get()
            src = req.src
            dst = req.dst
            part_nodes = req.nodes.to(self.device)
            # to local
            feature = self.feature[part_nodes]
            # to global
            resp = FeatureResponse(src, dst, feature)
            self.sync.response_queues[src].put(resp)
        for i in range(self.comm.ws - 1):
            resp = self.sync.response_queues[self.comm.rank].get()
            src = resp.src
            dst = resp.dst
            feature = resp.features
            feature_results[dst] = feature.to(self.device)
        total_features = []
        for feature in feature_results:
            total_features.append(feature)
        feature_beg = time.time()
        total_features = torch.cat(total_features)
        feature_end = time.time()
        total_features = total_features[feature_reorder]  #= total_features
        # if self.comm.rank == 0:
        #     print(f'feature cat {feature_end - feature_beg}')
        return total_features

    def __call__(self, rank):
        self.prepare(rank, *self.args)
        dataloader = torch.utils.data.DataLoader(self.train_idx,
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 drop_last=True)
        for i in range(self.num_epoch):
            count = 0
            for j in range(num_batch):
                count += 1
                src = self.comm.global_rank
                sample = self.sync.upstream_queues[src].get()
                batch_size, n_id, adjs, features = sample

                self.optimizer.zero_grad()
                out = self.model(features, adjs)
                label_ids = n_id[:batch_size].to(self.device)
                loss = F.nll_loss(out, self.y[label_ids].to(self.device))
                loss.backward()
                self.optimizer.step()
                if count >= num_batch:
                    break


class TrainerWrapper:
    def __init__(self, comm, sync):
        self.proc = TrainerProcess(comm, sync)

    def __call__(self, rank, *args):
        self.proc(*args)


def get_csr_from_coo(edge_index):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    node_count = max(np.max(src), np.max(dst))
    data = np.zeros(dst.shape, dtype=np.int32)
    csr_mat = csr_matrix(
        (data, (edge_index[0].numpy(), edge_index[1].numpy())))
    return csr_mat


if __name__ == '__main__':
    mp.set_start_method('spawn')
    ws = 1
    num_epoch = 1
    num_batch = 100
    batch_size = 128
    sizes = [15, 10, 5]
    root = "/home/dalong/data/"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-products')
    data = dataset[0]

    csr_mat = get_csr_from_coo(data.edge_index)

    train_idx = split_idx['train']
    edge_index = data.edge_index
    csr_mat = get_csr_from_coo(edge_index)
    x, y = data.x.share_memory_(), data.y.squeeze().share_memory_()
    sample_data = csr_mat, batch_size, sizes, train_idx
    train_data = dataset.num_features, 256, dataset.num_classes, 3, y
    comm = CommConfig(0, ws)
    sync = SyncManager(ws)
    proc = SingleProcess(num_epoch, num_batch, sample_data, train_data, x,
                         sync, comm)
    procs = launch_multiprocess(proc, ws)
    time.sleep(50)
    for p in procs:
        p.kill()
'''

def get_csr_from_coo(edge_index):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    node_count = max(np.max(src), np.max(dst))
    data = np.zeros(dst.shape, dtype=np.int32)
    csr_mat = csr_matrix((data, (edge_index[0].numpy(), edge_index[1].numpy())), shape=(node_count, node_count))
    return csr_mat
'''
