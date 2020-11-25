import asyncio
import concurrent
import copy
import os
import time
from typing import List, NamedTuple, Optional, Tuple

from quiver.coro.task import *

import torch
import torch_quiver as qv


class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size)


async def async_process(pool, func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(pool, func, *args)


class LayerSampleTask(TaskNode):
    def __init__(self, context, rank, quiver, batch, size, pool):
        super().__init__(context)
        self.rank = rank
        self.quiver = quiver
        self.batch = batch
        self.size = size
        self.pool = pool

    def set_batch(self, batch):
        self.batch = batch

    def get_request(self):
        return 'gpu'

    def _work(self):
        result, row_idx, col_idx = self.quiver.sample_sub(
            self.batch, self.size)
        self.result = result
        row_idx, col_idx = col_idx, row_idx
        edge_index = torch.stack([row_idx, col_idx], dim=0)

        size = torch.LongTensor([
            result.size(0),
            self.batch.size(0),
        ])

        e_id = torch.tensor([])  # e_id is not used in the example
        adj = Adj(edge_index, e_id, size)
        return result, adj

    async def do_work(self):
        return await async_process(self.loop, self._work)

    async def after_work(self):
        for child in self._children:
            child.set_batch(self.result)

    async def merge_result(self, me, children):
        if self.rank == -1:
            n_id, adj = me
            adjs = []
            return n_id, adjs
        n_id, adjs = children[0]
        _, adj = me
        adjs.insert(0, adj)
        return n_id, adjs


class CudaNeighborSampler(torch.utils.data.DataLoader):
    def __init__(self,
                 edge_index: torch.Tensor,
                 sizes: List[int],
                 node_idx: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 **kwargs):

        N = int(edge_index.max() + 1) if num_nodes is None else num_nodes
        # print('building quiver')
        t0 = time.time()
        edge_id = torch.zeros(edge_index.size(1), dtype=torch.long)
        self.quiver = qv.new_quiver_from_edge_index(N, edge_index, edge_id)
        d = time.time() - t0
        # print('build quiver took %fms' % (d * 1000))

        if node_idx is None:
            node_idx = torch.arange(N)
        elif node_idx.dtype == bool:
            node_idx = node_idx.nonzero().view(-1)

        self.sizes = sizes
        self.tasks = []
        self.context = TaskContext(0, GPUInfo(1))
        self.pool = concurrent.futures.ThreadPoolExecutor()
        self.build_tasks()

        super(CudaNeighborSampler, self).__init__(node_idx.tolist(),
                                                  collate_fn=self.sample,
                                                  **kwargs)

    def build_tasks(self):
        for i in range(len(self.sizes)):
            rank = -1 if i == len(self.sizes) - 1 else i
            task = LayerSampleTask(
                self.context, rank, self.quiver, None, self.sizes[i], self.pool)
            self.tasks.append(task)

        for i in range(len(self.sizes) - 1):
            self.tasks[i].add_child(self.tasks[i + 1])

    def sample(self, batch):
        t0 = time.time()
        ret = self._sample(batch)
        d = time.time() - t0
        # print('sample took %fms' % (d * 1000))
        return ret

    def _await_sample(self, batch):
        return asyncio.run(async_process(self.pool, self._sample, batch))

    def _coro_sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        adjs: List[Adj] = []

        n_id = batch

        self.tasks[0].set_batch(batch)
        n_id, adjs = asyncio.run(self.tasks[0].run())

        if len(adjs) > 1:
            return batch_size, n_id, adjs[::-1]
        else:
            return batch_size, n_id, adjs[0]

    def _sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        adjs: List[Adj] = []

        n_id = batch
        for size in self.sizes:
            result, row_idx, col_idx = self.quiver.sample_sub(n_id, size)
            assert (row_idx.max() < n_id.size(0))

            row_idx, col_idx = col_idx, row_idx
            edge_index = torch.stack([row_idx, col_idx], dim=0)

            size = torch.LongTensor([
                result.size(0),
                n_id.size(0),
            ])
            assert (row_idx.max() < result.size(0))
            assert (col_idx.max() < n_id.size(0))

            # print('size: %s' % (size))
            # FIXME: also sample e_id
            e_id = torch.tensor([])  # e_id is not used in the example
            adjs.append(Adj(edge_index, e_id, size))
            n_id = result

        if len(adjs) > 1:
            return batch_size, n_id, adjs[::-1]
        else:
            return batch_size, n_id, adjs[0]

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)
