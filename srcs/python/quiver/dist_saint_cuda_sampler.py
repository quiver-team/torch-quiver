import torch_quiver as qv
from torch_sparse import SparseTensor
from typing import Optional, List, NamedTuple, Optional, Tuple
from torch.distributed import rpc
from torch_geometric.data import GraphSAINTSampler
import torch


def sample_n(nodes, size):
    neighbors, counts = None, None

    return neighbors, counts

def subgraph_nodes_n(nodes, i):
    row, col, edge_index = None, None, None
    return row, col, edge_index

class Comm:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

sample_nodes = sample_n
subgraph_nodes = subgraph_nodes_n

class distributeCudaRWSampler(GraphSAINTSampler):
    r"""The GraphSAINT random walk sampler class (see
    :class:`torch_geometric.data.GraphSAINTSampler`).
    Args:
        walk_length (int): The length of each random walk.
    """
    def __init__(self,
                 comm,
                 graph,
                 feature_func,
                 device,
                 batch_size: int,
                 walk_length: int,
                 num_steps: int = 1,
                 sample_coverage: int = 0,
                 save_dir: Optional[str] = None,
                 log: bool = True,
                 **kwargs):
        self.comm = comm
        self.walk_length = walk_length
        data, local2global, global2local, node2rank = graph
        self.local2global = local2global
        self.global2local = global2local
        self.node2rank = node2rank
        self.node_feature = feature_func
        super(distributeCudaRWSampler,
              self).__init__(data, batch_size, num_steps, sample_coverage,
                             save_dir, log, **kwargs)
        self.cuda_device = torch.device('cuda:' + str(device))
        self.adj = self.adj.to(self.cuda_device)

    @property
    def __filename__(self):
        return (f'{self.__class__.__name__.lower()}_{self.walk_length}_'
                f'{self.sample_coverage}.pt')

    def get_data(self, n_id, is_feature):
        ranks = self.node2rank(n_id)
        input_orders = torch.arange(n_id.size(0), dtype=torch.long)
        reorder = torch.empty_like(input_orders)
        res = []
        beg = 0
        for i in range(self.comm.world_size):
            mask = torch.eq(ranks, i)
            part_nodes = torch.masked_select(n_id, mask)
            part_orders = torch.masked_select(input_orders, mask)
            if part_nodes.size(0) >= 1:
                if i == self.comm.rank:
                    local_nodes = part_nodes
                    res.append(torch.LongTensor([]))
                else:
                    res.append(
                        rpc.rpc_async(f"worker{i}",
                                      self.node_feature,
                                      args=(part_nodes, is_feature),
                                      kwargs=None,
                                      timeout=-1.0))
                nodes = part_nodes
                reorder[beg:beg + part_nodes.size(0)] = part_orders
                beg += part_nodes.size(0)
            else:
                res.append(torch.LongTensor([]))
        if local_nodes is not None:
            nodes = self.global2local(local_nodes)
            if is_feature:
                local_res = self.x[nodes]
            else:
                local_res = self.y[nodes]
            res[self.comm.rank] = local_res
        for i in range(len(res)):
            if not isinstance(res[i], torch.Tensor):
                res[i] = res[i].wait()
        res = torch.cat(res)
        origin_res = torch.empty_like(res)
        origin_res[reorder] = res
        return origin_res

    def __sample_nodes__(self, batch_size):
        # make start nodes
        start = torch.randint(0, self.N, (batch_size,), dtype=torch.long)
        ranks = self.node2rank(start)
        local_nodes = None
        res = []
        node_indices = []
        acc = [start]
        for step in range(self.walk_length):
            for i in range(self.comm.world_size):
                # for every device check how many nodes
                mask = torch.eq(ranks, i)
                part_nodes = torch.masked_select(start, mask)
                # nodes as the the current, pointer ordered inputs to accumulate the partial nodes
                if part_nodes.size(0) >= 1:
                    # if current server then local
                    if i == self.comm.rank:
                        local_nodes = part_nodes
                        res.append(torch.LongTensor([]))
                    # remote server
                    else:
                        res.append(
                            rpc.rpc_async(f"worker{i}",
                                          sample_nodes,
                                          args=(part_nodes, 1),
                                          kwargs=None,
                                          timeout=-1.0))

                else:
                    res.append(torch.LongTensor([]))
            # local server has nodes
            if local_nodes is not None:
                nodes = self.global2local(local_nodes)
                nodes = nodes.to(self.cuda_device)
                # walk length in current step is 1
                node_idx = self.adj.random_walk(nodes, 1)[:, 1]
                res[self.comm.rank] = node_idx.to(torch.device('cpu'))
            for i in range(len(res)):
                # if not isinstance(res[i], tuple):
                if not isinstance(res[i], torch.Tensor):
                    res[i] = res[i].wait()
                return_nodes = res[i]
                node_indices.append(return_nodes)
            if step < self.walk_length - 1:
                start = torch.cat(node_indices)
                ranks = self.node2rank(start)
                acc.append(start)
                node_indices = []
                res = []
                local_nodes = None
            else:
                acc.append(torch.cat(node_indices))
                acc = torch.cat(acc)
        return acc

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
                    futures.append((torch.LongTensor([]), torch.LongTensor([]), torch.LongTensor([])))
                # remote server
                else:
                    futures.append(
                        rpc.rpc_async(f"worker{i}",
                                      subgraph_nodes,
                                      args=(part_nodes, 1),
                                      kwargs=None,
                                      timeout=-1.0))

            else:
                futures.append((torch.LongTensor([]), torch.LongTensor([]), torch.LongTensor([])))
        # local server has nodes
        if local_nodes is not None:
            nodes = self.global2local(local_nodes)
            nodes = nodes.to(self.cuda_device)
            row, col, edge_index = qv.saint_subgraph(nodes, adj_rowptr, adj_row, adj_col)
            futures[self.comm.rank] = row.to(cpu), col.to(cpu), edge_index.to(cpu)

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

        out = SparseTensor(row = ret_row,
                           rowptr = None,
                           col= ret_cols,
                           value = ret_vals,
                           sparse_sizes=(node_idx.size(0), node_idx.size(0)),
                           is_sorted=False)
        return out, ret_edgeindex

    def __getitem__(self, idx):
        node_idx = self.__sample_nodes__(self.__batch_size__).unique()
        adj, _ = self.__cuda_saint_subgraph__(node_idx)
        return node_idx, adj
