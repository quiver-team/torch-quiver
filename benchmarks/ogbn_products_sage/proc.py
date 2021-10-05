import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from quiver.async_cuda_sampler import AsyncCudaNeighborSampler
from quiver.async_data import DataManager
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

DEVICE_NUM = 2
BUFFER_SIZE = 2


class InputRequest:
    def __init__(self, index, nodes, size, train_device):
        self.index = index
        self.nodes = nodes
        self.size = size
        self.train_device = train_device


class SampleRequest:
    def __init__(self, index, src, dst, nodes):
        self.index = index
        self.src = src
        self.dst = dst
        self.nodes = nodes


class SampleResponse:
    def __init__(self, index, src, dst, nodes, reindex_results):
        self.index = index
        self.src = src
        self.dst = dst
        self.nodes = nodes
        self.reindex_results = reindex_results


# class ReindexRequest:
#     def __init__(self, index, src, dst, inputs, outputs, counts):
#         self.index = index
#         self.src = src
#         self.dst = dst
#         self.inputs = inputs
#         self.outputs = outputs
#         self.counts = counts

# class ReindexResponse:
#     def __init__(self, index, src, dst, nodes, row, col):
#         self.index = index
#         self.src = src
#         self.dst = dst
#         self.nodes = nodes
#         self.row = row
#         self.col = col


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
    def __init__(self, local_reindex, sampler):
        self.local_reindex = local_reindex
        self.sampler = sampler


class CommConfig:
    def __init__(self,
                 global_rank,
                 group_rank,
                 group_size,
                 group_peers,
                 upstream_peer,
                 downstream_peer,
                 other_peers=None):
        self.global_rank = global_rank
        self.group_rank = group_rank
        self.group_size = group_size
        self.group_peers = group_peers
        self.upstream_peer = upstream_peer
        self.downstream_peer = downstream_peer
        self.other_peers = other_peers


class SyncManager:
    def __init__(self, num_proc, init_size):
        # TODO: Init properly
        self.request_queues = [(mp.Queue(init_size)) for i in range(num_proc)]
        self.response_queues = [(mp.Queue(init_size)) for i in range(num_proc)]
        self.upstream_queues = [(mp.Queue(init_size)) for i in range(num_proc)]
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

    def run(self, *args):
        pass


class CudaSamplerProcess(QuiverProcess):
    def prepare(self, data):
        device, edge_index, sizes = data
        self.sizes = sizes
        self.loader = AsyncCudaNeighborSampler(edge_index, device=device)

    def run(self, *args):
        upstream_queue = self.sync.upstream_queues[self.comm.global_rank]
        response_queue = self.sync.response_queues[self.comm.upstream_peer]
        while True:
            while not upstream_queue.empty():
                req = upstream_queue.get()
                # print(f'sampler {self.comm.group_rank} reindex')
                nodes = req.nodes
                batch_size = len(nodes)
                reindex_results = []
                for size in self.sizes:
                    out, cnt = self.loader.sample_layer(nodes, size)
                    frontier, row_idx, col_idx = self.loader.reindex(
                        nodes, out, cnt)
                    reindex_results.append((len(frontier), col_idx, row_idx))
                    nodes = frontier
                response_queue.put(
                    SampleResponse(req.index, req.src, req.dst, nodes,
                                   reindex_results))


class MicroBatchIndex:
    def __init__(self, batch_id, rank, size):
        self.batch_id = batch_id
        self.rank = rank
        self.size = size


class DataProcess(QuiverProcess):
    def prepare(self, data):
        manager, feature_size, buffer_size = data
        self.manager = manager
        self.manager.prepare()
        self.feature_size = feature_size
        self.buffer_size = buffer_size

    def run(self, *args):
        upstream_queue = self.sync.upstream_queues[self.comm.global_rank]
        request_queue = self.sync.request_queues[self.comm.global_rank]
        response_queue = self.sync.response_queues[self.comm.global_rank]
        while True:
            while not request_queue.empty():
                req = request_queue.get()
                # print(f'data {self.comm.group_rank} feature request')
                self.handle_feature_request(req)
            while not response_queue.empty():
                res = response_queue.get()
                if isinstance(res, SampleResponse):
                    # print(f'data {self.comm.group_rank} sample response')
                    self.handle_sample_response(res)
                # elif isinstance(res, ReindexResponse):
                # print(f'data {self.comm.group_rank} reindex response')
                # self.handle_reindex_response(res)
                else:
                    # print(f'data {self.comm.group_rank} feature response')
                    self.handle_feature_response(res)
            if len(self.manager.buffers
                   ) < self.buffer_size and not upstream_queue.empty():
                req = upstream_queue.get()
                self.handle_input_request(req)

    def handle_input_request(self, input_request):
        # print(f'data {self.comm.group_rank} a batch')
        train_device = input_request.train_device
        nodes = input_request.nodes.to(self.manager.sample_device)
        batch_id = input_request.index.batch_id
        size = input_request.size
        sampler = self.comm.other_peers.sampler
        self.manager.init_entry(nodes, batch_id, size, train_device)
        index = MicroBatchIndex(batch_id, 0, 1)
        req = SampleRequest(index, self.comm.global_rank, sampler, nodes)
        self.sync.upstream_queues[sampler].put(req)

    def handle_sample_response(self, sample_response):
        batch_id = sample_response.index.batch_id
        nodes = sample_response.nodes.clone().to(self.manager.device)
        ret = self.manager.recv_sample(batch_id, nodes,
                                       sample_response.reindex_results)
        reorder, res = self.manager.dispatch(nodes, self.feature_size)
        res = self.manager.prepare_request(batch_id, reorder, res)
        group_peers = self.comm.group_peers
        for rank, nodes in enumerate(res):
            if rank == self.comm.group_rank:
                features = self.manager.feature[nodes]
                self.manager.recv_feature(batch_id, rank, len(res), features)
                continue
            index = MicroBatchIndex(batch_id, rank, len(res))
            req = FeatureRequest(index, self.comm.global_rank,
                                 group_peers[rank], nodes)
            self.sync.request_queues[group_peers[rank]].put(req)

    # def handle_reindex_response(self, reindex_response):
    #     batch_id = reindex_response.index.batch_id
    #     nodes = reindex_response.nodes.clone()
    #     row = reindex_response.row.clone()
    #     col = reindex_response.col.clone()
    #     ret = self.manager.recv_reindex(batch_id, nodes, row, col)
    #     nodes, temp_layer = ret
    #     if temp_layer >= 0:
    #         sampler = self.comm.other_peers.sampler
    #         index = MicroBatchIndex(batch_id, 0, 1)
    #         req = SampleRequest(
    #             index, self.comm.global_rank, sampler, nodes, self.sizes[temp_layer])
    #         self.sync.request_queues[sampler].put(req)
    #     else:
    #         reorder, res = self.manager.dispatch(
    #             nodes, self.feature_size)
    #         res = self.manager.prepare_request(batch_id, reorder, res)
    #         group_peers = self.comm.group_peers
    #         for rank, nodes in enumerate(res):
    #             if rank == self.comm.group_rank:
    #                 features = self.manager.feature[nodes]
    #                 self.manager.recv_feature(
    #                     batch_id, rank, len(res), features)
    #                 continue
    #             index = MicroBatchIndex(batch_id, rank, len(res))
    #             req = FeatureRequest(
    #                 index, self.comm.global_rank, group_peers[rank], nodes)
    #             self.sync.request_queues[group_peers[rank]].put(req)

    def handle_feature_request(self, feature_request):
        index = feature_request.index
        nodes = feature_request.nodes
        src = feature_request.src
        dst = feature_request.dst
        features = self.manager.feature[nodes]
        response = FeatureResponse(index, src, dst, features)
        self.sync.response_queues[src].put(response)

    def handle_feature_response(self, feature_response):
        batch_id = feature_response.index.batch_id
        features = feature_response.features
        rank = feature_response.index.rank
        size = feature_response.index.size
        features = self.manager.recv_feature(batch_id, rank, size, features)
        if features is not None:
            batch_size, n_ids, adjs = self.manager.prepare_train(batch_id)
            args = (batch_size, n_ids, adjs, features)
            trainer = self.comm.downstream_peer
            # while True:
            #     if self.sync.upstream_queues[trainer].empty():
            self.sync.upstream_queues[trainer].put(args)
            # break
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
        total_time = 0
        for i in range(epoch):
            count = 0
            t0 = time.time()
            for j in range(num_batch):
                count += 1
                src = self.comm.global_rank
                queue_beg = time.time()
                sample = self.sync.upstream_queues[src].get()
                queue_end = time.time()
                batch_size, n_id, adjs, features = sample

                self.optimizer.zero_grad()
                label_ids = n_id[:batch_size].to(self.device)
                out = self.model(features, adjs)
                loss = F.nll_loss(out, self.y[label_ids].to(self.device))
                loss.backward()
                self.optimizer.step()
                dur = time.time() - t0
                if count > 10:
                    total_time += dur
                    if count % 10 == 0 and self.comm.global_rank == 2 * DEVICE_NUM:
                        print(
                            f'rank {self.comm.group_rank} total {count} avg {total_time / (count - 10)}'
                        )
                        print(
                            f'trainer {self.comm.global_rank} wait {queue_end - queue_beg} a batch'
                        )
                t0 = time.time()
                if count >= num_batch:
                    break


class TrainerWrapper:
    def __init__(self, comm, sync, *args):
        self.proc = TrainerProcess(comm, sync)
        self.args = args

    def __call__(self, rank):
        self.proc.comm.global_rank += rank
        self.proc.comm.group_rank = rank
        self.proc(*self.args)


def identity(x, y, z):
    return x


def rank_func(x):
    return torch.fmod(x, DEVICE_NUM)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    # global_rank, group_rank, group_size, group_peers,
    #              downstream_peer, other_peers
    num_epoch = 1
    num_batch = 4000
    batch_size = 128

    sampler_size = DEVICE_NUM
    trainer_size = DEVICE_NUM
    data_size = DEVICE_NUM
    global_size = sampler_size + data_size + trainer_size
    global_rank = 0
    sample_comms = []
    data_comms = []

    home = os.getenv('HOME')
    data_dir = osp.join(home, '.pyg')
    root = osp.join(data_dir, 'data', 'products')
    root = "/home/dalong/data/"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-products')
    graph_data = dataset[0]
    feature = graph_data.x.share_memory_()

    train_idx = split_idx['train']

    for i in range(sampler_size):
        sample_comm = CommConfig(global_rank + i, i, sampler_size,
                                 list(range(sampler_size)), i + sampler_size,
                                 -1)
        sample_comms.append(sample_comm)
    global_rank += sampler_size
    for i in range(data_size):
        other_peers = DataPeerInfo(i, i)
        data_comm = CommConfig(
            global_rank + i, i, data_size,
            list(range(sampler_size, sampler_size + data_size)), i,
            i + sampler_size + data_size, other_peers)
        data_comms.append(data_comm)
    global_rank += data_size
    train_comm = CommConfig(
        global_rank, 0, trainer_size,
        list(
            range(sampler_size + data_size,
                  sampler_size + data_size + trainer_size)), -1, -1)

    sync = SyncManager(global_size, 8)
    sample_procs = []
    data_procs = []
    for sample_comm in sample_comms:
        sample = CudaSamplerProcess(sample_comm, sync)
        sample_proc = mp.Process(target=sample,
                                 args=((sample_comm.group_rank,
                                        graph_data.edge_index, [15, 10, 5]),
                                       num_epoch, num_batch))
        sample_proc.start()
        sample_procs.append(sample_proc)
    for data_comm in data_comms:
        data = DataProcess(data_comm, sync)
        device = data_comm.group_rank
        feature_devices = list(range(data_size))
        feature_to_local = identity
        feature_rank = rank_func
        manager = DataManager(device, feature, device, feature_devices,
                              feature_to_local, feature_rank)
        data_proc = mp.Process(target=data,
                               args=((manager, data_size, BUFFER_SIZE),
                                     num_epoch, num_batch))
        data_proc.start()
        data_procs.append(data_proc)

    train = TrainerWrapper(train_comm, sync,
                           (dataset.num_features, 256, dataset.num_classes, 3,
                            graph_data.y.squeeze()), num_epoch, num_batch)
    train_procs = launch_multiprocess(train, trainer_size)
    train_idx = train_idx.repeat(20)
    dataloader = torch.utils.data.DataLoader(train_idx,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             drop_last=True)

    dst_data = 0
    count = 0
    sync.beg_barrier.wait()
    print('procs beg')
    for nodes in dataloader:
        index = MicroBatchIndex(count, 0, 1)
        req = InputRequest(index, nodes, data_size, dst_data)
        sync.upstream_queues[dst_data + sampler_size].put(req)
        dst_data = (dst_data + 1) % data_size
        count += 1
        if count >= 16000:
            break
    time.sleep(50)

    # sampler_proc = mp.Process(target=sample, args=(
    #     (0, data.edge_index, train_idx, [15, 10, 5], 512), 1, 385))
    # trainer_proc = mp.Process(target=train, args=(
    #     (dataset.num_features, 256, dataset.num_classes, 3, data.x, data.y), 1, 385))
    # sampler_proc.start()
    # trainer_proc.start()
    # sync.beg_barrier.wait()
    # print('begin')
    # t0 = time.time()
    # sync.end_barrier.wait()
    # print(f'total took {time.time() - t0}')
