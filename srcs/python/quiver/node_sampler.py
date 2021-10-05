import copy
from typing import List, Optional, Tuple, NamedTuple, Union, Callable

import torch
from torch import Tensor
from torch_sparse import SparseTensor
import time
import torch_quiver as qv


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
            print('in shuffle')
        return iter(self.n_ids)

    def __len__(self):
        return self.num_parts


class RandomNodeCudaSampler(torch.utils.data.DataLoader):
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
                 data,
                 device,
                 num_parts: int,
                 shuffle: bool = False,
                 **kwargs):
        assert data.edge_index is not None

        self.N = N = data.num_nodes
        self.E = data.num_edges
        self.cuda_device = torch.device('cuda:' + str(device))
        self.adj = SparseTensor(row=data.edge_index[0],
                                col=data.edge_index[1],
                                value=torch.arange(
                                    self.E, device=data.edge_index.device),
                                sparse_sizes=(N, N)).to(self.cuda_device)

        self.data = copy.copy(data)
        self.data.edge_index = None
        self.deg_out = None

        super(RandomNodeCudaSampler,
              self).__init__(self,
                             batch_size=1,
                             sampler=RandomIndexSampler(
                                 self.N, num_parts, shuffle),
                             collate_fn=self.__collate__,
                             **kwargs)

    def __getitem__(self, idx):
        return idx

    def __collate__(self, node_idx):
        node_idx = node_idx[0]
        data = self.data.__class__()
        data.num_nodes = node_idx.size(0)
        row, col, value = self.adj.coo()
        node_idx = node_idx.to(self.cuda_device)
        rowptr = self.adj.storage.rowptr()
        if (self.deg_out is None):
            self.deg_out = self.adj.storage.rowcount()
        deg = torch.index_select(self.deg_out, 0, node_idx)
        subgraph = qv.saint_subgraph(node_idx, rowptr, row, col, deg)
        row, col, edge_index = subgraph
        if value is not None:
            value = value[edge_index]

        adj = SparseTensor(row=row,
                           rowptr=None,
                           col=col,
                           value=value,
                           sparse_sizes=(node_idx.size(0), node_idx.size(0)),
                           is_sorted=True)
        row, col, edge_idx = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)

        for key, item in self.data:
            if isinstance(item, Tensor) and item.size(0) == self.N:
                data[key] = item[node_idx]
            elif isinstance(item, Tensor) and item.size(0) == self.E:
                data[key] = item[edge_idx]
            else:
                data[key] = item

        return data


class RandomNodeSampler(torch.utils.data.DataLoader):
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
    def __init__(self, data, num_parts: int, shuffle: bool = False, **kwargs):
        assert data.edge_index is not None

        self.N = N = data.num_nodes
        self.E = data.num_edges

        self.adj = SparseTensor(row=data.edge_index[0],
                                col=data.edge_index[1],
                                value=torch.arange(
                                    self.E, device=data.edge_index.device),
                                sparse_sizes=(N, N))

        self.data = copy.copy(data)
        self.data.edge_index = None

        super(RandomNodeSampler, self).__init__(self,
                                                batch_size=1,
                                                sampler=RandomIndexSampler(
                                                    self.N, num_parts,
                                                    shuffle),
                                                collate_fn=self.__collate__,
                                                **kwargs)

    def __getitem__(self, idx):
        return idx

    def __collate__(self, node_idx):
        node_idx = node_idx[0]

        data = self.data.__class__()
        data.num_nodes = node_idx.size(0)
        adj, _ = self.adj.saint_subgraph(node_idx)
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
