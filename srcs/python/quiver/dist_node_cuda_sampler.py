import copy
from typing import List, Optional, Tuple, NamedTuple, Union, Callable

import torch
from torch import Tensor
from torch_sparse import SparseTensor
import time
import torch_quiver as qv
from torch.distributed import rpc


def subgraph_nodes_n(nodes, i):
    row, col, edge_index = None, None, None
    return row, col, edge_index


class Comm:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size


subgraph_nodes = subgraph_nodes_n


class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)


class RandomIndexSampler(torch.utils.data.Sampler):
    def __init__(self, num_nodes: int, num_parts: int, shuffle: bool = False):
        self.N = num_nodes
        self.num_parts = num_parts
        self.shuffle = shuffle
        self.n_ids = self.get_node_indices()

    def get_node_indices(self):
        n_id = torch.randint(self.num_parts, (self.N, ), dtype=torch.long)
        n_ids = [(n_id == i).nonzero(as_tuple=False).view(-1)
                 for i in range(self.num_parts)]
        return n_ids

    def __iter__(self):
        if self.shuffle:
            self.n_ids = self.get_node_indices()
        return iter(self.n_ids)

    def __len__(self):
        return self.num_parts


class distributeCudaRandomNodeSampler(torch.utils.data.DataLoader):
    r"""A data loader that randomly samples nodes within a graph and returns
    their induced subgraph.

    .. note::

        For an example of using :obj:`RandomNodeSampler`, see
        `examples/ogbn_proteins_deepgcn.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        ogbn_proteins_deepgcn.py>`_.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        num_parts (int): The number of partitions.
        shuffle (bool, optional): If set to :obj:`True`, the data is reshuffled
            at every epoch (default: :obj:`False`).
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`num_workers`.
    """
    def __init__(self,
                 comm,
                 graph,
                 feature_func,
                 device,
                 num_parts: int,
                 shuffle: bool = False,
                 **kwargs):
        self.comm = comm
        data, local2global, global2local, node2rank = graph
        self.local2global = local2global
        self.global2local = global2local
        self.node2rank = node2rank
        self.node_feature = feature_func
        self.cuda_device = torch.device('cuda:' + str(device))

        assert data.edge_index is not None

        self.N = N = data.num_nodes
        self.E = data.num_edges
        self.adj = SparseTensor(row=data.edge_index[0],
                                col=data.edge_index[1],
                                value=torch.arange(
                                    self.E, device=data.edge_index.device),
                                sparse_sizes=(N, N)).to(self.cuda_device)
        self.data = copy.copy(data)
        self.data.edge_index = None

        super(distributeCudaRandomNodeSampler,
              self).__init__(self,
                             batch_size=1,
                             sampler=RandomIndexSampler(
                                 self.N, num_parts, shuffle),
                             collate_fn=self.__collate__,
                             **kwargs)
        self.deg_out = self.adj.storage.rowcount()

    def __getitem__(self, idx):
        return idx

    def __cuda_saint_subgraph__(
            self, node_idx: torch.Tensor) -> Tuple[SparseTensor, torch.Tensor]:
        rows = []
        cols = []
        edge_indices = []
        # splite node idx
        ranks = self.node2rank(node_idx)
        local_nodes = None
        futures = []
        adj_row, adj_col, adj_value = self.adj.coo()
        adj_rowptr = self.adj.storage.rowptr()
        cpu = torch.device('cpu')

        for i in range(self.comm.world_size):
            # for every device check how many nodes on the device
            mask = torch.eq(ranks, i)
            part_nodes = torch.masked_select(node_idx, mask)
            # nodes as the the current, pointer ordered inputs to accumulate the partial nodes
            if part_nodes.size(0) >= 1:
                # if current server then local
                if i == self.comm.rank:
                    local_nodes = part_nodes
                    futures.append((torch.LongTensor([]), torch.LongTensor([]),
                                    torch.LongTensor([])))
                # remote server
                else:
                    futures.append(
                        rpc.rpc_async(f"worker{i}",
                                      subgraph_nodes,
                                      args=(part_nodes, 1),
                                      kwargs=None,
                                      timeout=-1.0))

            else:
                futures.append((torch.LongTensor([]), torch.LongTensor([]),
                                torch.LongTensor([])))
        # local server has nodes
        if local_nodes is not None:
            nodes = self.global2local(local_nodes)
            nodes = nodes.to(self.cuda_device)

            deg = torch.index_select(self.deg_out, 0, nodes)
            row, col, edge_index = qv.saint_subgraph(nodes, adj_rowptr,
                                                     adj_row, adj_col, deg)

            row = row.to(cpu)
            col = col.to(cpu)
            edge_index = edge_index.to(cpu)

            futures[self.comm.rank] = row, col, edge_index

        for i in range(len(futures)):
            if not isinstance(futures[i], tuple):
                futures[i] = futures[i].wait()
            row, col, edge_index = futures[i]
            rows.append(row)
            cols.append(col)
            edge_indices.append(edge_index)

        ret_row = torch.cat(rows)
        ret_cols = torch.cat(cols)
        ret_edgeindex = torch.cat(edge_indices)

        if adj_value is not None:
            ret_vals = adj_value[ret_edgeindex].to(cpu)
        out = SparseTensor(row=ret_row,
                           rowptr=None,
                           col=ret_cols,
                           value=ret_vals,
                           sparse_sizes=(node_idx.size(0), node_idx.size(0)),
                           is_sorted=False)
        return out, ret_edgeindex

    def __collate__(self, node_idx):
        node_idx = node_idx[0]
        data = self.data.__class__()
        data.num_nodes = node_idx.size(0)
        node_idx = node_idx.unique()
        adj, _ = self.__cuda_saint_subgraph__(node_idx)
        row, col, edge_idx = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)
        data.node_idx = node_idx
        data.train_mask = self.data.train_mask[node_idx]
        for key, item in self.data:
            if isinstance(item, Tensor) and item.size(0) == self.N:
                data[key] = item[node_idx]
            elif isinstance(item, Tensor) and item.size(0) == self.E:
                data[key] = item[edge_idx]
            else:
                data[key] = item
        return data
