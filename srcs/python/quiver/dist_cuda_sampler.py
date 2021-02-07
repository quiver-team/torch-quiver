import torch
from torch.distributed import rpc

import torch_quiver as qv

from typing import List, NamedTuple, Optional, Tuple


def sample_n(nodes, size):
    neighbors, counts = None, None

    return neighbors, counts

sample_neighbor = sample_n

class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size)


class Comm:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size


class SyncDistNeighborSampler(torch.utils.data.DataLoader):
    def __init__(self, comm, graph, train_idx, layer_sizes, device, **kwargs):
        torch.set_num_threads(1)
        self.comm = comm
        self.sizes = layer_sizes
        N, edge_index, edge_id, local2global, global2local, node2rank = graph
        self.quiver = qv.new_quiver_from_edge_index(
            N, edge_index, edge_id, device)
        self.local2global = local2global
        self.global2local = global2local
        self.node2rank = node2rank

        super(SyncDistNeighborSampler, self).__init__(train_idx.tolist(),
                                                      collate_fn=self.sample,
                                                      **kwargs)

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        adjs: List[Adj] = []

        n_id = batch
        for size in self.sizes:
            ranks = self.node2rank(n_id)
            local_nodes = None
            futures = []
            inputs = []
            for i in range(self.comm.world_size):
                mask = torch.eq(ranks, i)
                part_nodes = torch.masked_select(n_id, mask)
                if part_nodes.size(0) >= 1:
                    if i == self.comm.rank:
                        local_nodes = part_nodes
                        inputs.insert(0, local_nodes)
                    else:
                        futures.append(rpc.rpc_async(f"worker{i}", sample_neighbor, args=(
                            part_nodes, size), kwargs=None, timeout=-1.0))
                        inputs.append(part_nodes)
            res = []
            if local_nodes is not None:
                nodes = self.global2local(local_nodes)
                neighbors, counts = self.quiver.sample_neighbor(0, nodes, size)
                neighbors = self.local2global(neighbors)
                res.append((neighbors, counts))
            for f in futures:
                res.append(f.wait())
            result, row_idx, col_idx = self.quiver.reindex_group(
                0, inputs, res)
            size = torch.LongTensor([
                result.size(0),
                n_id.size(0),
            ])
            row_idx, col_idx = col_idx, row_idx
            edge_index = torch.stack([row_idx, col_idx], dim=0)
            e_id = torch.tensor([])
            adjs.append(Adj(edge_index, e_id, size))
            n_id = result
        if len(adjs) > 1:
            return batch_size, n_id, adjs[::-1]
        else:
            return batch_size, n_id, adjs[0]
