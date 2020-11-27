import asyncio
import time


class TaskNode:
    def __init__(self, context):
        self._children = []
        self._context = context
        self._request = None
        self._resource = None

    def add_child(self, child):
        self._children.append(child)

    def get_request(self):
        return self._request

    def get_resource(self):
        return self._resource

    async def request(self):
        self._resource = await self._context.request(self.get_request())

    async def revoke(self):
        await self._context.revoke(self.get_resource())

    async def do_work(self):
        pass

    async def after_work(self):
        pass

    async def merge_result(self, me, children):
        pass

    async def run(self):
        await self.request()
        my_result = await self.do_work()
        await self.revoke()
        await self.after_work()
        tasks = []
        for child in self._children:
            task = asyncio.create_task(child.run())
            tasks.append(task)
        children_results = []
        for task in tasks:
            result = await task
            children_results.append(result)
        result = await self.merge_result(my_result, children_results)
        return result


class SleepTask(TaskNode):
    def __init__(self, context):
        super().__init__(context)

    def get_request(self):
        return 'gpu'

    async def do_work(self):
        await asyncio.sleep(1)
        print('sleep')


class DemoTask(TaskNode):
    def __init__(self, context, fanout):
        super().__init__(context)
        for i in range(fanout):
            super().add_child(SleepTask(context))

    def get_request(self):
        return 'gpu'

    async def do_work(self):
        await asyncio.sleep(1)
        print('demo sleep')
