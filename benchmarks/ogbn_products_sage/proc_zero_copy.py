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
from scipy.sparse import csr_matrix
import numpy as np


class SampleRequest:
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


class TrainArgs:
    def __init__(self, batch_size, n_ids, adjs, features):
        self.batch_size = batch_size
        self.n_ids = n_ids
        self.adjs = adjs
        self.features = features


class DataPeerInfo:
    def __init__(self, local_reindex, global_samplers):
        self.local_reindex = local_reindex
        self.global_samplers = global_samplers


class CommConfig:
    def __init__(self, global_rank, group_rank, group_size, group_peers,
                 downstream_peer, other_peers=None):
        self.global_rank = global_rank
        self.group_rank = group_rank
        self.group_size = group_size
        self.group_peers = group_peers
        self.downstream_peer = downstream_peer
        self.other_peers = other_peers


class SyncManager:
    def __init__(self, num_proc, init_size):
        # TODO: Init properly
        self.request_queues = [(mp.Queue(init_size))] * num_proc
        self.response_queues = [(mp.Queue(init_size))] * num_proc
        self.upstream_queues = [(mp.Queue(init_size))] * num_proc
        self.downstream_queues = [(mp.Queue(init_size))] * num_proc
        self.beg_barrier = mp.Barrier(num_proc + 1)
        self.end_barrier = mp.Barrier(num_proc + 1)


class QuiverProcess:
    def __init__(self, comm, sync):
        self.comm = comm
        self.sync = sync

    def prepare(self, data):
        pass

    def __call__(self, data, *args):
        self.prepare(data)
        self.sync.beg_barrier.wait()
        self.run(*args)
        self.sync.end_barrier.wait()

    def run(self, *args):
        pass


class CudaSamplerProcess(QuiverProcess):
    def prepare(self, data):
        device, csr_mat, train_idx, sizes, batch_size = data
        indptr = torch.from_numpy(csr_mat.indptr)
        indices = torch.from_numpy(csr_mat.indices)
        self.loader = AsyncCudaNeighborSampler(csr_indptr=indptr,
                                               csr_indices=indices,
                                               device=device,
                                               node_idx=train_idx,
                                               sizes=sizes,
                                               batch_size=batch_size,
                                               shuffle=True)

    def run(self, *args):
        upstream_queue = self.sync.upstream_queues[self.comm.global_rank]
        request_queue = self.sync.request_queues[self.comm.global_rank]
        response_queue = self.sync.response_queues[self.comm.global_rank]
        downstream_queue = self.sync.downstream_queues[self.comm.global_rank]
        while True:
            while not request_queue.empty():
                req = request_queue.get()
                nodes, counts = self.loader.sample_layer(req.nodes, req.size)
                response_queue.put(SampleResponse(
                    req.index, req.src, req.dst, nodes, counts))
            while not upstream_queue.empty():
                req = upstream_queue.get()
                nodes, row, col = self.loader.reindex(
                    req.reorder, req.inputs, req.outputs, req.counts)
                downstream_queue.put(ReindexResponse(
                    req.index, req.src, req.dst, nodes, row, col))


class MicroBatchIndex:
    def __init__(self, batch_id, rank, size):
        self.batch_id = batch_id
        self.rank = rank
        self.size = size


class DataProcess(QuiverProcess):
    def prepare(self, data):
        manager, sample_size, feature_size, sizes = data
        self.manager = manager
        self.sample_size = sample_size
        self.feature_size = feature_size
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
            request = ReindexRequest(
                index, self.global_rank, reindex_peer, reorder, inputs, outputs, counts)
            self.sync.upstream_queues[reindex_peer].put(request)

    def handle_reindex_response(self, reindex_response):
        batch_id = reindex_response.index.batch_id
        nodes = reindex_response.nodes
        row = reindex_response.row
        col = reindex_response.col
        ret = self.manager.recv_reindex(batch_id, nodes, row, col)
        nodes, temp_layer = ret
        if temp_layer >= 0:
            reorder, res = self.manager.dispatch(
                nodes, self.sample_size, False)
            res = self.manager.prepare_request(batch_id, reorder, res, False)
            global_samplers = self.comm.other_peers.global_samplers
            for rank, nodes in enumerate(res):
                index = MicroBatchIndex(batch_id, rank, len(res))
                req = SamplerRequest(
                    index, self.comm.global_rank, global_samplers[rank], nodes, self.sizes[temp_layer])
                self.sync.request_queues[global_samplers[rank]].put(req)
        else:
            reorder, res = self.manager.dispatch(
                nodes, self.feature_size, True)
            res = self.manager.prepare_request(batch_id, reorder, res, True)
            group_peers = self.comm.group_peers
            for rank, nodes in enumerate(res):
                index = MicroBatchIndex(batch_id, rank, len(res))
                req = FeatureRequest(
                    index, self.comm.global_rank, group_peers[rank], nodes)
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
        device = torch.device('cuda:' +
                              str(self.comm.group_rank) if torch.cuda.is_available() else 'cpu')
        self.device = device
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

    def run(self, *args):
        epoch, num_batch = args
        for i in range(epoch):
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
    csr_mat = csr_matrix((data, (edge_index[0].numpy(), edge_index[1].numpy())), shape=(node_count, node_count))
    return csr_mat
    

if __name__ == '__main__':
    mp.set_start_method('spawn')
    sample_comm = CommConfig(0, 0, 1, [], 1)
    train_comm = CommConfig(1, 0, 1, [], [])
    sync = SyncManager(2, 32)
    sample = SamplerProcess(sample_comm, sync)
    train = TrainerProcess(train_comm, sync)
    home = os.getenv('HOME')
    data_dir = osp.join(home, '.pyg')
    root = osp.join(data_dir, 'data', 'products')
    dataset = PygNodePropPredDataset('ogbn-products', root)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-products')
    data = dataset[0]
    
    csr_mat = get_csr_from_coo(data.edge_index)
    
    train_idx = split_idx['train']
    sampler_proc = mp.Process(target=sample, args=(
        (0, csr_mat, train_idx, [15, 10, 5], 512), 1, 385))
    trainer_proc = mp.Process(target=train, args=(
        (dataset.num_features, 256, dataset.num_classes, 3, data.x, data.y), 1, 385))
    sampler_proc.start()
    trainer_proc.start()
    sync.beg_barrier.wait()
    print('begin')
    t0 = time.time()
    sync.end_barrier.wait()
    print(f'total took {time.time() - t0}')
