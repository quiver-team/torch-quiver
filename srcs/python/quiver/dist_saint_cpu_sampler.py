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
class distributeRWSampler(GraphSAINTSampler):
    def __init__(self,
                 comm,
                 graph,
                 feature_func,
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

        self.walk_length = walk_length
        super(distributeRWSampler,
              self).__init__(data, batch_size, num_steps, sample_coverage,
                             save_dir, log, **kwargs)

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

    def __sample__nodes__(self, batch_size):
        # make start nodes
        start = torch.randint(0, self.N, (batch_size,), dtype=torch.long)
        ranks = self.node2rank(start)
        local_nodes = None
        res = []
        ordered_inputs = []
        ordered_outputs = []
        # input_orders = torch.arange(start.size(0), dtype=torch.long)
        for i in range(self.comm.world_size):
            # for every device check how many nodes
            mask = torch.eq(ranks, i)
            part_nodes = torch.masked_select(start, mask)
            # input orders which order it is input?
            # part_orders = torch.masked_select(input_orders, mask)
            # nodes as the the current, pointer ordered inputs to accumulate the partial nodes
            nodes = torch.LongTensor([])
            if part_nodes.size(0) >= 1:
                # if current server then local
                if i == self.comm.rank:
                    local_nodes = part_nodes
                    res.append(
                        (torch.LongTensor([]), torch.LongTensor([])))
                # remote server
                else:
                    res.append(
                        rpc.rpc_async(f"worker{i}",
                                      sample_nodes,
                                      args=(nodes, batch_size),
                                      kwargs=None,
                                      timeout=-1.0))

                    # nodes = part_nodes
                    # reorder[beg:beg + part_nodes.size(0)] = part_orders
                    # where to begin this is the server
                    # beg += part_nodes.size(0)
            else:
                # no  nodes at i at all
                res.append((torch.LongTensor([]), torch.LongTensor([])))
            # result append
            # ordered_inputs.append(nodes)
        # ordered_inputs = torch.cat(ordered_inputs)
        # local server has nodes
        if local_nodes is not None:
            nodes = self.global2local(local_nodes)
            # neighbors, counts = self.quiver.sample_neighbor(nodes, size)
            node_idx = self.adj.random_walk(nodes, self.walk_length)
            node_idx = self.local2global(node_idx)
            res[self.comm.rank] = node_idx
        for i in range(len(res)):
            if not isinstance(res[i], torch.Tensor):
                res[i] = res[i].wait()
            return_nodes = res[i].view(-1)
            ordered_outputs.append(return_nodes)
        ordered_outputs = torch.cat(ordered_outputs)
        # ordered_counts = torch.cat(ordered_counts)
        # result, row_idx, col_idx = self.quiver.reindex_group(
        #     ordered_inputs, n_id, ordered_counts, ordered_outputs)
        #     size = torch.LongTensor([
        #         result.size(0),
        #         n_id.size(0),
        #     ])
        #     row_idx, col_idx = col_idx, row_idx
        #     edge_index = torch.stack([row_idx, col_idx], dim=0)
        #     e_id = torch.tensor([])
        #     adjs.append(Adj(edge_index, e_id, size))
        #     n_id = result
        # if len(adjs) > 1:
        #     return batch_size, n_id, adjs[::-1]
        # else:
        return ordered_outputs

    # def __saint__subgraph__(self, node_idx):
    #     'split the node idx as list of indices and separate them'
    #     ranks = self.node2rank(node_idx)
    #     local_nodes = None
    #     res = []
    #     # ordered_inputs = []
    #     outputs = []
    #     input_orders = torch.arange(node_idx.size(0), dtype=torch.long)
    #     for i in range(self.comm.world_size):
    #         # for every device check how many nodes
    #         mask = torch.eq(ranks, i)
    #         part_nodes = torch.masked_select(node_idx, mask)
    #         # input orders which order it is input?
    #         # part_orders = torch.masked_select(input_orders, mask)
    #         # nodes as the the current, pointer ordered inputs to accumulate the partial nodes
    #         nodes = torch.LongTensor([])
    #         if part_nodes.size(0) >= 1:
    #             # if current server then local
    #             if i == self.comm.rank:
    #                 local_nodes = part_nodes
    #                 res.append(
    #                     (torch.LongTensor([]), torch.LongTensor([])))
    #             # remote server
    #             else:
    #                 res.append(
    #                     rpc.rpc_async(f"worker{i}",
    #                                   sample_nodes,
    #                                   args=(nodes, start, batch_size),
    #                                   kwargs=None,
    #                                   timeout=-1.0))
    #
    #                 nodes = part_nodes
    #                 # reorder[beg:beg + part_nodes.size(0)] = part_orders
    #                 # where to begin this is the server
    #                 # beg += part_nodes.size(0)
    #         else:
    #             # no  nodes at i at all
    #             res.append((torch.LongTensor([]), torch.LongTensor([])))
    #         # result append
    #         # ordered_inputs.append(nodes)
    #     # ordered_inputs = torch.cat(ordered_inputs)
    #     # local server has nodes
    #     if local_nodes is not None:
    #         nodes = self.global2local(local_nodes)
    #         # neighbors, counts = self.quiver.sample_neighbor(nodes, size)
    #         # node_idx = self.adj.random_walk(nodes, self.walk_length)
    #         node_idx =self.adj.saint_subgraph(nodes)
    #         node_idx = self.local2global(node_idx)
    #         res[self.comm.rank] = node_idx
    #     for i in range(len(res)):
    #         if not isinstance(res[i], torch.Tensor):
    #             res[i] = res[i].wait()
    #         adj, _ = res[i]
    #         outputs.append()
    #     outputs = torch.cat(outputs)
    #     # ordered_counts = torch.cat(ordered_counts)
    #     # result, row_idx, col_idx = self.quiver.reindex_group(
    #     #     ordered_inputs, n_id, ordered_counts, ordered_outputs)
    #     #     size = torch.LongTensor([
    #     #         result.size(0),
    #     #         n_id.size(0),
    #     #     ])
    #     #     row_idx, col_idx = col_idx, row_idx
    #     #     edge_index = torch.stack([row_idx, col_idx], dim=0)
    #     #     e_id = torch.tensor([])
    #     #     adjs.append(Adj(edge_index, e_id, size))
    #     #     n_id = result
    #     # if len(adjs) > 1:
    #     #     return batch_size, n_id, adjs[::-1]
    #     # else:
    #     return outputs