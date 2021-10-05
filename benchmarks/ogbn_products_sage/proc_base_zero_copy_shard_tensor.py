import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

import torch_quiver as qv
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


class SampleRequest:
    def __init__(self, src, dst, nodes, size):
        self.src = src
        self.dst = dst
        self.nodes = nodes
        self.size = size


class BatchSampleResponse:
    def __init__(self, index, src, dst, outputs, counts):
        self.index = index
        self.src = src
        self.dst = dst
        self.outputs = outputs
        self.counts = counts


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

    def prepare(self, rank, sample_data, train_data, shard_tensor_item_ipc,
                sync, comm):
        #####################################################################################################
        # Bind Task To NUMA Node And Sleep 1s So That Next Time This Processing Is Runing On Target NUMA Node
        #####################################################################################################
        total_nodes = info.get_max_node() + 1
        current_node = 0 if rank < 2 else 1
        schedule.bind(current_node)

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
        num_features, num_hidden, num_classes, num_layers, y = train_data
        self.y = y
        device = torch.device(
            'cuda:' +
            str(self.comm.rank) if torch.cuda.is_available() else 'cpu')
        self.device = device

        ###################################
        # Rebuild Tensor In Child Process
        ###################################
        self.feature = qv.ShardTensor(rank)
        for ipc_item in shard_tensor_item_ipc:
            item = qv.ShardTensorItem()
            item.from_ipc(ipc_item)
            self.feature.append(item)

        torch.cuda.set_device(device)

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
        #########################
        # Collect By Shard Tensor
        #########################
        nodes = nodes.to(self.device)
        total_features = self.feature[nodes]
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
                    print(f'rank {self.comm.rank} sample {t1 - t0}')
                    print(f'rank {self.comm.rank} feature {t2 - t1}')
                    print(f'rank {self.comm.rank} took {time.time() - t0}')
                    t0 = time.time()
                    if count >= self.num_batch:
                        cont = False
                        break
        # print(f'rank {self.comm.rank} sample avg {self.sample_time}')
        # print(f'rank {self.comm.rank} reindex avg {self.reindex_time}')
        # print(f'rank {self.comm.rank} request avg {self.request_time}')
        # print(f'rank {self.comm.rank} handle avg {self.handle_time}')
        # print(f'rank {self.comm.rank} recv avg {self.recv_time}')
        # print(f'rank {self.comm.rank} queue avg {self.queue_time}')


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
    data = dataset[0]
    train_idx = split_idx['train']
    edge_index = data.edge_index
    csr_mat = get_csr_from_coo(edge_index)
    y = data.y.squeeze().share_memory_()

    ######################################
    # Init Shard Tensor In Main Process
    ######################################
    qv.init_p2p()
    shard_tensors = []
    shard_tensor = qv.ShardTensor(0)
    half_count = data.x.shape[0] // 2
    shard_tensor.append(data.x[:half_count], 0)
    shard_tensor.append(data.x[half_count:], 1)
    shard_tensor_ipc = shard_tensor.share_ipc()
    shard_item_ipc = [item.share_ipc() for item in shard_tensor_ipc]

    sample_data = csr_mat, batch_size, sizes, train_idx
    train_data = dataset.num_features, 256, dataset.num_classes, 3, y
    comm = CommConfig(0, ws)
    sync = SyncManager(ws)
    proc = SingleProcess(num_epoch, num_batch, sample_data, train_data,
                         shard_item_ipc, sync, comm)
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
