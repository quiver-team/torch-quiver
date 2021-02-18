from quiver.coro.dataloader import AsyncDataGenerator, AsyncDataLoader
from quiver.coro.task_context import TaskContext

import asyncio
import concurrent
import random
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch_quiver as qv


class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size)


class CudaDataset:
    def __init__(self, edge_index, sizes, node_idx):
        N = int(edge_index.max() + 1)
        edge_attr = torch.arange(edge_index.size(1))
        edge_id = torch.zeros(edge_index.size(1), dtype=torch.long)
        self.quiver = qv.new_quiver_from_edge_index(N, edge_index, edge_id)
        if node_idx is None:
            node_idx = torch.arange(N)
        self.node_idx = node_idx.tolist()
        self.sizes = sizes


class CudaNeighborGenerator(AsyncDataGenerator):
    def prepare(self, dataset):
        edge_index, sizes, node_idx = dataset
        self.dataset = CudaDataset(edge_index, sizes, node_idx)
        self.context = TaskContext(1, self.num_worker)
        # torch.cuda.set_device(self.rank)
        # self.stream_pool = qv.StreamPool(self.num_worker)
        # self.dataset.quiver.set_pool(self.stream_pool)

    async def async_run(self, batch):
        typ, num = await self.context.request({'gpu': 1})
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.pool, self._sample, batch,
                                            num)
        await self.context.revoke((typ, num))
        return result

    def _sample(self, batch, stream):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        torch.cuda.set_device(self.rank)

        adjs: List[Adj] = []

        n_id = batch
        for size in self.dataset.sizes:
            result, row_idx, col_idx = self.dataset.quiver.sample_sub(
                stream, n_id, size)
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

    def next_batch(self):
        if self.index >= len(self.dataset.node_idx):
            return
        beg = self.index
        end = self.index + self.batch_size
        if end > len(self.dataset.node_idx):
            end = len(self.dataset.node_idx)
        self.index = end
        return self.dataset.node_idx[beg:end]

    def shuffle(self):
        random.shuffle(self.dataset.node_idx)


class CudaNeighborLoader(AsyncDataLoader):
    def __init__(self, dataset, batch_size, num_worker):
        super().__init__(dataset, batch_size, num_worker)
        _, _, node_idx = dataset
        self.len = len(node_idx) // batch_size + 1

    def __len__(self):
        return self.len

    def new_generator(self, dataset, batch_size, num_worker, queue, rank=0):
        return CudaNeighborGenerator(dataset, batch_size, num_worker, queue,
                                     rank)
