import os
import os.path as osp

import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from scipy.sparse import csr_matrix
from quiver.async_cuda_sampler import AsyncCudaNeighborSampler

from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
import time
from quiver.shard_tensor import ShardTensor as PyShardTensor
from quiver.shard_tensor import ShardTensorConfig, DeviceCollectionJob

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import random
from typing import List, NamedTuple, Optional, Tuple
from torch.utils.tensorboard import SummaryWriter

REINDEX = True


def reindex_by_config(adj_csr, graph_feature, gpu_portion):
    degree = adj_csr.indptr[1:] - adj_csr.indptr[:-1]
    degree = torch.from_numpy(degree)
    node_count = degree.shape[0]
    _, prev_order = torch.sort(degree, descending=True)
    total_range = torch.arange(node_count, dtype=torch.long)
    perm_range = torch.randperm(int(node_count * gpu_portion))

    new_order = torch.zeros_like(prev_order)
    prev_order[:int(node_count * gpu_portion)] = prev_order[perm_range]
    new_order[prev_order] = total_range
    graph_feature = graph_feature[prev_order]
    return graph_feature, new_order


class InterProcData:
    edge_index = None
    train_idx = None
    val_idx = None
    test_idx = None
    y = None
    num_features = None
    num_classes = None
    new_order = None


class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size)


class NodeDataset(torch.utils.data.IterableDataset):
    def __init__(self, nodes, batch_size, tensor_offset_device):
        super().__init__()
        self.tensor_offset_device = tensor_offset_device
        self.nodes = nodes
        self.batch_size = batch_size

    def next(self):
        start = 0
        input_orders = torch.arange(self.batch_size, dtype=torch.long)

        while start < len(self.nodes):
            if len(self.nodes) - start < self.batch_size:
                break
            nodes = self.nodes[start:start + self.batch_size]
            dispatch_book = {}
            for device in self.tensor_offset_device:
                request_nodes_mask = (
                    nodes >= self.tensor_offset_device[device].start) & (
                        nodes < self.tensor_offset_device[device].end)
                if (request_nodes_mask.shape[0] > 0):
                    request_nodes = torch.masked_select(
                        nodes, request_nodes_mask)
                    part_orders = torch.masked_select(input_orders,
                                                      request_nodes_mask)
                    dispatch_book[device] = DeviceCollectionJob(
                        part_orders, request_nodes)
            yield nodes, dispatch_book
            start += self.batch_size

    def __len__(self):
        return len(self.nodes) // self.batch_size

    def __iter__(self):
        random.shuffle(self.nodes)
        return self.next()


class SAGE(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers=2):
        super(SAGE, self).__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


def get_csr_from_coo(edge_index):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    node_count = max(np.max(src), np.max(dst))
    data = np.zeros(dst.shape, dtype=np.int32)
    csr_mat = csr_matrix(
        (data, (edge_index[0].numpy(), edge_index[1].numpy())))
    return csr_mat


def sample(sampler, device, input_nodes, sizes):
    nodes = input_nodes.to(device)
    adjs = []

    batch_size = len(nodes)
    for size in sizes:
        out, cnt = sampler.sample_layer(nodes, size)
        frontier, row_idx, col_idx = sampler.reindex(nodes, out, cnt)
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


def fake_collate(x):
    return x


def run(rank, world_size, shard_tensor_ipc_handle,
        inter_proc_data: InterProcData):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    train_idx = inter_proc_data.train_idx
    y = inter_proc_data.y.squeeze().to(rank)

    new_order = inter_proc_data.new_order.to(rank)

    csr_mat = get_csr_from_coo(inter_proc_data.edge_index)
    sampler = AsyncCudaNeighborSampler(csr_indptr=csr_mat.indptr,
                                       csr_indices=csr_mat.indices,
                                       device=rank,
                                       copy=True)
    x = PyShardTensor.new_from_share_ipc(shard_tensor_ipc_handle, rank)
    batch_size = 2048

    train_loader = torch.utils.data.DataLoader(train_idx,
                                               batch_size=batch_size,
                                               pin_memory=True)
    writer = SummaryWriter(f'{rank}_pyg_log')

    torch.manual_seed(12345)
    model = SAGE(inter_proc_data.num_features,
                 256,
                 inter_proc_data.num_classes,
                 num_layers=3).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    time_points = []
    iter_points = []

    for epoch in range(1, 21):
        model.train()
        start_time = time.time()
        for iter_step, seeds in enumerate(train_loader):
            start = time.time()
            n_id, batch_size, adjs = sample(sampler, rank, seeds, [15, 10, 5])
            writer.add_scalar(f"{rank}_sample", time.time() - start, iter_step)
            start = time.time()
            new_n_id = new_order[n_id]

            feature = x[new_n_id]
            writer.add_scalar(f"{rank}_feature",
                              time.time() - start, iter_step)

            adjs = [adj.to(rank) for adj in adjs]
            start = time.time()
            optimizer.zero_grad()
            out = model(feature, adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()
            writer.add_scalar(f"{rank}_train", time.time() - start, iter_step)
            if rank == 0 and iter_step > 10:
                iter_points.append(time.time() - start_time)
                writer.add_scalar(
                    f"{rank}_throughput", world_size * batch_size /
                    np.mean(np.array(iter_points[10:])), iter_step)

            start_time = time.time()
        time_points.clear()
        iter_points.clear()

        #if rank == 0:
        #    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    torch.cuda.set_device(0)
    home = os.getenv('HOME')
    data_dir = osp.join(home, '.pyg')
    root = osp.join(data_dir, 'data', 'products')
    dataset = PygNodePropPredDataset('ogbn-products', root)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']

    data = dataset[0]
    inter_proc_data = InterProcData()
    data.edge_index.share_memory_()

    inter_proc_data.edge_index = data.edge_index
    inter_proc_data.num_classes = dataset.num_classes
    inter_proc_data.num_features = dataset.num_features
    inter_proc_data.train_idx = train_idx

    inter_proc_data.y = data.y

    shard_tensor_config = ShardTensorConfig({0: "200M"})

    shard_tensor = PyShardTensor(0, shard_tensor_config)
    csr_mat = get_csr_from_coo(inter_proc_data.edge_index)
    if REINDEX:
        feature, new_order = reindex_by_config(csr_mat, data.x, 0)
    else:
        feature = data.x
        node_count = csr_mat.indptr.shape[0] - 1
        new_order = torch.arange(node_count, dtype=torch.long)

    inter_proc_data.new_order = new_order

    shard_tensor.from_cpu_tensor(data.x)
    ipc_handle = shard_tensor.share_ipc()
    world_size = 1
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(run,
             args=(world_size, ipc_handle, inter_proc_data),
             nprocs=world_size,
             join=True)
