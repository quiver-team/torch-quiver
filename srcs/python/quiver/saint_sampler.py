from torch_geometric.data import GraphSAINTSampler
from torch_geometric.data import GraphSAINTRandomWalkSampler
import torch
# from torch.distributed import rpc
import random
import torch_quiver as qv
from torch_sparse import SparseTensor
from typing import Optional, List, NamedTuple, Tuple
import time
import os.path as osp
from torch_geometric.data import Data


class quiverRWSampler(GraphSAINTRandomWalkSampler):
    @property
    def __filename__(self):
        hardcode = "GraphSAINTRandomWalkSampler"
        return (f'{hardcode.lower()}_{self.walk_length}_'
                f'{self.sample_coverage}.pt')

    def __collate__(self, data_list):
        assert len(data_list) == 1
        node_idx, adj = data_list[0]
        # get the node_idx and adj

        data = Data()
        data.num_nodes = node_idx.size(0)
        row, col, edge_idx = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)
        data.train_mask = self.data.train_mask[node_idx]

        if self.sample_coverage > 0:
            data.node_norm = self.node_norm[node_idx]
            data.edge_norm = self.edge_norm[edge_idx]

        return data, node_idx


class CudaRWSampler(GraphSAINTSampler):
    r"""The GraphSAINT random walk sampler class (see
    :class:`torch_geometric.data.GraphSAINTSampler`).
    Args:
        walk_length (int): The length of each random walk.
    """
    def __init__(self,
                 data,
                 device,
                 batch_size: int,
                 walk_length: int,
                 num_steps: int = 1,
                 sample_coverage: int = 0,
                 save_dir: Optional[str] = None,
                 log: bool = True,
                 **kwargs):
        self.walk_length = walk_length
        self.cuda_device = torch.device('cuda:' + str(device))
        self.deg_out = None
        super(CudaRWSampler,
              self).__init__(data, batch_size, num_steps, sample_coverage,
                             save_dir, log, **kwargs)

    @property
    def __filename__(self):
        hardcode = "GraphSAINTRandomWalkSampler"
        return (f'{hardcode.lower()}_{self.walk_length}_'
                f'{self.sample_coverage}.pt')

    def __sample_nodes__(self, batch_size):
        start = torch.randint(0, self.N, (batch_size, ), dtype=torch.long)
        start = start.to(self.cuda_device)
        # start_t  = time.time()
        if not self.adj.storage.col().is_cuda:
            print("set device")
            self.adj = self.adj.to(self.cuda_device)
        node_idx = self.adj.random_walk(start.flatten(), self.walk_length)
        end = time.time()
        # print("rw", end - start_t)
        return node_idx.view(-1)

    def __cuda_saint_subgraph__(
            self, node_idx: torch.Tensor) -> Tuple[SparseTensor, torch.Tensor]:
        row, col, value = self.adj.coo()
        rowptr = self.adj.storage.rowptr()
        # start_t = time.time()
        if (self.deg_out is None):
            print("calculating")
            self.deg_out = self.adj.storage.rowcount()
        deg = torch.index_select(self.deg_out, 0, node_idx)
        data = qv.saint_subgraph(node_idx, rowptr, row, col, deg)

        row, col, edge_index = data

        if value is not None:
            value = value[edge_index]

        out = SparseTensor(row=row,
                           rowptr=None,
                           col=col,
                           value=value,
                           sparse_sizes=(node_idx.size(0), node_idx.size(0)),
                           is_sorted=True)
        return out, edge_index

    def __getitem__(self, idx):
        # start = time.time()
        node_idx = self.__sample_nodes__(self.__batch_size__)
        # end = time.time()
        # print("TOTAL RW takes : ", end - start)
        # start = time.time()
        node_idx = node_idx.unique()
        # end = time.time()
        # print("unique takes : ", end - start)
        # print(node_idx)
        # start = time.time()
        adj, _ = self.__cuda_saint_subgraph__(node_idx)
        # end = time.time()
        # print("TOTAL subgraph takes : ", end - start)
        return node_idx, adj

    def __collate__(self, data_list):
        assert len(data_list) == 1
        node_idx, adj = data_list[0]
        # get the node_idx and adj

        data = Data()
        data.num_nodes = node_idx.size(0)
        row, col, edge_idx = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)
        data.train_mask = self.data.train_mask[node_idx]

        if self.sample_coverage > 0:
            data.node_norm = self.node_norm[node_idx]
            data.edge_norm = self.edge_norm[edge_idx]

        return data, node_idx


class QuiverSAINTEdgeSampler(GraphSAINTSampler):
    r"""The GraphSAINT edge sampler class (see
    :class:`torch_geometric.data.GraphSAINTSampler`).
    """
    def __init__(self,
                 data,
                 batch_size: int,
                 num_steps: int = 1,
                 sample_coverage: int = 0,
                 save_dir: Optional[str] = None,
                 log: bool = True,
                 **kwargs):
        super(QuiverSAINTEdgeSampler,
              self).__init__(data, batch_size, num_steps, sample_coverage,
                             save_dir, log, **kwargs)
        self.deg_in = 1. / self.adj.storage.colcount()
        self.deg_out = 1. / self.adj.storage.rowcount()

    def __sample_nodes__(self, batch_size):
        row, col, _ = self.adj.coo()
        i = 0
        batch_count = 0

        source_node_sample = []
        target_node_sample = []
        while i < self.E:
            if batch_count == batch_size:
                break
            u = row[i]
            v = col[i]
            prob = (self.deg_in[u]) + (self.deg_out[v])
            rand = random.uniform(0, 1)
            if rand > prob:
                i = i + 1
                continue
            else:
                batch_count = batch_count + 1
                i = i + 1
                source_node_sample.append(u)
                target_node_sample.append(v)
        return torch.cat([
            torch.LongTensor(source_node_sample),
            torch.LongTensor(target_node_sample)
        ], -1)
