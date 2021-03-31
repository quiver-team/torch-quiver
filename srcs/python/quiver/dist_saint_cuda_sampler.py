import torch_quiver as qv
from torch_sparse import SparseTensor
from typing import Optional, List, NamedTuple, Optional, Tuple
from torch.distributed import rpc
from torch_geometric.data import GraphSAINTSampler
import torch


def sample_n(nodes, size):
    neighbors, counts = None, None

    return neighbors, counts

class Comm:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

sample_nodes = sample_n
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
        print("after init")

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
        # cols = []
        # rows = []
        # values = []
        # input_orders = torch.arange(start.size(0), dtype=torch.long)
        for step in range(self.walk_length):
            for i in range(self.comm.world_size):
                # for every device check how many nodes
                mask = torch.eq(ranks, i)
                part_nodes = torch.masked_select(start, mask)
                # input orders which order it is input?
                # part_orders = torch.masked_select(input_orders, mask)
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
                                          args=(part_nodes, self.walk_length),
                                          kwargs=None,
                                          timeout=-1.0))

                        # nodes = part_nodes
                        # reorder[beg:beg + part_nodes.size(0)] = part_orders
                        # where to begin this is the server
                        # beg += part_nodes.size(0)s
                else:
                    # no  nodes at i at all
                    # res.append((torch.LongTensor([]),
                    #             torch.LongTensor([]),
                    #             torch.LongTensor([]),
                    #             torch.LongTensor([])))
                    res.append(torch.LongTensor([]))
            # local server has nodes

            if local_nodes is not None:
                nodes = self.global2local(local_nodes)
                nodes = nodes.to(self.cuda_device)
                # walk length in current step is 1
                node_idx = self.adj.random_walk(nodes, 1).view(-1)
                res[self.comm.rank] = node_idx.to(torch.device('cpu'))
            del nodes
            del node_idx
            torch.cuda.empty_cache()
            for i in range(len(res)):
                # if not isinstance(res[i], tuple):
                if not isinstance(res[i], torch.Tensor):
                    res[i] = res[i].wait()
                return_nodes = res[i]
                node_indices.append(return_nodes)

            node_indices = torch.cat(node_indices)
            if step < self.walk_length - 1:
                start = node_indices
                ranks = self.node2rank(start)
                del node_indices
                node_indices = []
                torch.cuda.empty_cache()
        return node_indices

    def __cuda_saint_subgraph__(
            self, node_idx: torch.Tensor) -> Tuple[SparseTensor, torch.Tensor]:
        row, col, value = self.adj.coo()
        rowptr = self.adj.storage.rowptr()

        data = qv.saint_subgraph(node_idx, rowptr, row, col)
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
        node_idx = self.__sample_nodes__(self.__batch_size__).unique()
        adj, _ = self.__cuda_saint_subgraph__(node_idx.to(self.cuda_device))
        return node_idx, adj.to(torch.device('cpu'))
