import asyncio
import concurrent
import multiprocessing as mp
import time
import horovod.torch as hvd
import torch


class AsyncDataGenerator:
    def __init__(self, dataset, batch_size, num_worker, queue, rank=0):
        self.rank = rank
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.index = 0
        self.consumed = 0
        self.round = 0
        self.pool = concurrent.futures.ThreadPoolExecutor()
        self.prepare(dataset)
        self.queue = queue
        self.tasks = [self.create_task() for i in range(num_worker)]

    def prepare(self, dataset):
        self.dataset = dataset

    def next_batch(self):
        if self.index >= len(self.dataset):
            return
        beg = self.index
        end = self.index + self.batch_size
        if end > len(self.dataset):
            end = len(self.dataset)
        self.index = end
        return self.dataset[beg:end]

    def reset(self):
        self.shuffle()
        self.index = 0
        self.consumed = 0
        self.round = 0
        self.tasks = [self.create_task() for i in range(self.num_worker)]

    def shuffle(self):
        return

    def create_task(self):
        batch = self.next_batch()
        if batch is None:
            return
        task = asyncio.create_task(self.async_run(batch))
        return task

    async def async_run(self, batch):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.pool, self.sample, batch)
        return result

    def sample(self, batch):
        return batch

    async def get_once(self):
        if self.consumed < self.index:
            i = self.round
            if self.tasks[i] is None:
                return
            result = await self.tasks[i]
            self.tasks[i] = self.create_task()
            self.consumed += self.batch_size
            self.round = (i + 1) % self.num_worker
            return result


class AsyncDataLoader:
    def __init__(self, dataset, batch_size, num_worker):
        self.queue = mp.Queue(num_worker)
        rank = 0
        try:
            rank = hvd.local_rank()
        except ValueError:
            pass
        proc = mp.Process(target=self.sample_process,
                          args=(rank, dataset, batch_size, num_worker,
                                self.queue))
        proc.start()
        self.proc = proc

    async def async_generate_process(self,
                                     dataset,
                                     batch_size,
                                     num_worker,
                                     queue,
                                     rank=0):
        sampler = self.new_generator(dataset, batch_size, num_worker, queue,
                                     rank)
        while True:
            batch = await sampler.get_once()
            while batch is not None:
                self.queue.put(batch)
                batch = await sampler.get_once()
            self.queue.put(None)
            time.sleep(1)
            cont = self.queue.get()
            if cont is True:
                sampler.reset()
            else:
                return

    def sample_process(self, rank, dataset, batch_size, num_worker, queue):
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
        asyncio.run(
            self.async_generate_process(dataset, batch_size, num_worker, queue,
                                        rank))

    def new_generator(self, dataset, batch_size, num_worker, queue, rank=0):
        return AsyncDataGenerator(dataset, batch_size, num_worker, queue, rank)

    def __iter__(self):
        return self

    def __next__(self):
        result = self.queue.get()
        if result is None:
            raise StopIteration
        return result

    def reset(self):
        self.queue.put(True)
        time.sleep(1)

    def close(self):
        self.queue.put(False)
        time.sleep(1)
