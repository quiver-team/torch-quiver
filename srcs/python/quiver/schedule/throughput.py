import time


# throughput statistics of sample & train
class ThroughputStats:
    def __init__(self, num_warm=8, num_batch=64):
        self.num_warm = num_warm
        self.num_batch = num_batch

    def test(self, sampler, trainer):
        count = 0
        beg = 0.0
        end = 0.0
        for batch in sampler:
            count += 1
            if count == self.num_warm:
                beg = time.time()
            if count - self.num_warm == self.num_batch:
                end = time.time()
                break
            trainer.train(batch)
        return (end - beg) / self.num_batch


class SamplerChooser:
    def __init__(self, trainer, num_warm=8, num_batch=64):
        self.num_warm = num_warm
        self.num_batch = num_batch
        self.trainer = trainer

    def choose_sampler(self, samplers):
        cost = 999.999
        best = None
        for sampler in samplers:
            stats = ThroughputStats(self.num_warm, self.num_batch)
            result = stats.test(sampler, self.trainer)
            if result < cost:
                cost = result
                best = sampler
        return best
