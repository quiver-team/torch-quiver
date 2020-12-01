import asyncio
import concurrent
import multiprocessing as mp
import time


class AsyncDataGenerator:
    def __init__(self, dataset, batch_size, num_worker, queue):
        self.prepare(dataset)
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.index = 0
        self.consumed = 0
        self.round = 0
        self.pool = concurrent.futures.ThreadPoolExecutor()
        self.tasks = [self.create_task() for i in range(num_worker)]
        self.queue = queue

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
        return self.dataset[beg: end]

    def reset(self):
        self.shuffle()
        self.index = 0
        self.consumed = 0
        self.round = 0
        self.tasks = [self.create_task() for i in range(num_worker)]

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
        self.queue = mp.Queue()
        proc = mp.Process(target=self.sample_process, args=(
            dataset, batch_size, num_worker, self.queue))
        proc.start()
        self.proc = proc

    async def async_generate_process(self, dataset, batch_size, num_worker, queue):
        sampler = self.new_generator(
            dataset, batch_size, num_worker, queue)
        batch = await sampler.get_once()
        while batch is not None:
            self.queue.put(batch)
            batch = await sampler.get_once()
        self.queue.put(None)

    def sample_process(self, dataset, batch_size, num_worker, queue):
        asyncio.run(self.async_generate_process(
            dataset, batch_size, num_worker, queue))

    def new_generator(self, dataset, batch_size, num_worker, queue):
        return AsyncDataGenerator(dataset, batch_size, num_worker, queue)

    def __iter__(self):
        return self

    def __next__(self):
        result = self.queue.get()
        if result is None:
            raise StopIteration
        return result
