import asyncio
import random


class TaskContext:
    def __init__(self, cpu, gpu, prefer=0.9):
        self.cpu = cpu
        self.gpu = gpu
        self.prefer = prefer
        self.cores = asyncio.Queue()
        self.streams = asyncio.Queue()
        for i in range(cpu):
            self.cores.put_nowait(i)
        for i in range(gpu):
            self.streams.put_nowait(i)

    async def request(self, dic):
        if 'gpu' in dic and 'cpu' in dic:
            if random.random() < self.prefer:
                stream = await self.streams.get()
                return 'gpu', stream
            else:
                core = await self.cores.get()
                return 'cpu', core
        elif 'gpu' in dic:
            stream = await self.streams.get()
            return 'gpu', stream
        else:
            core = await self.cores.get()
            return 'cpu', core

    async def revoke(self, res):
        typ, num = res
        if typ == 'gpu':
            await self.streams.put(num)
        else:
            await self.cores.put(num)
