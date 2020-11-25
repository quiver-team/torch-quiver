import asyncio

class GPUInfo:
    def __init__(self, num):
        self._num_stream = num

    def get_num_stream(self):
        return self._num_stream

class TaskContext:
    def __init__(self, cpu, gpu):
        self.cpu = cpu
        self.gpu = gpu
        self.streams = asyncio.Queue()
        for i in range(gpu.get_num_stream()):
            self.streams.put_nowait(i)
        
    async def request(self, typ):
        if typ == 'gpu':
            stream = await self.streams.get()
            return stream
        else:
            return None
    
    async def revoke(self, stream):
        if stream is not None:
            stream = await self.streams.put(stream)
