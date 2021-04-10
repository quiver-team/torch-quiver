class Policy:
    def __init__(self, batch_size, num_dev, num_train, num_cpu):
        self.batch_size = batch_size // num_train
        self.num_dev = num_dev
        self.num_train = num_train
        self.num_group = 1
        self.num_sub = 0
        self.num_cpu = num_cpu
        if num_dev == num_train:
            part = [1] * num_dev
        else:
            part = [1] * (num_dev - num_train)
            self.num_sub = 1
        self.group_part = [part]

    def __str__(self):
        return f'train {self.num_train}, sub {self.num_sub}, group {self.num_group}, cpu {self.num_cpu}'

    def add_group(self):
        part = [1] * self.num_dev
        self.num_group += 1
        self.group_part.append(part)

    def add_sub_group(self):
        part = [1] * (self.num_dev - self.num_train)
        self.num_group += 1
        self.num_sub += 1
        self.group_part.insert(0, part)

    def remove_group(self):
        self.num_group -= 1
        self.group_part.pop()

    def remove_sub_group(self):
        self.num_group -= 1
        self.num_sub -= 1
        self.group_part.pop(0)

    def set_part(self, index, part):
        self.group_part[index] = part

    def get_part(self, index):
        return self.group_part[index]

    def count(self):
        return self.num_group

    def count_sub(self):
        return self.num_sub


class PolicyFilter:
    def __init__(self, batch_size, num_dev, num_cpu):
        self.batch_size = batch_size
        self.num_dev = num_dev
        self.num_cpu = num_cpu
        self.policies = [
            Policy(batch_size, num_dev, i, num_cpu)
            for i in range(1, num_dev + 1)
        ]
        self.index = 0
        self.finished = [False] * num_dev
        self.stats = [None] * num_dev
        self.first = True
        self.prev = 999.99, 0, 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.policies[self.index].count() - self.policies[
                self.index].count_sub() >= 3:
            self.finished[self.index] = True
        if self.finished[self.index]:
            a, b, c = self.prev
            a *= 0.95
            b *= 0.95
            c *= 0.95
            if self.index >= self.num_dev or self.stats[self.index] > (a, b,
                                                                       c):
                raise StopIteration
            self.prev = self.stats[self.index]
            self.index += 1
        elif not self.first:
            self.policies[self.index].add_group()
        self.first = False
        return self.policies[self.index]

    def set_stats(self, stats):
        if self.stats[self.index] is None:
            self.stats[self.index] = stats
        else:
            if stats > self.stats[self.index]:
                self.finished[self.index] = True
                self.policies[self.index].remove_group()
            else:
                self.stats[self.index] = stats

    def best_group(self):
        ret = 999.99, 1, 1
        index = 0
        for i in range(self.num_dev):
            if self.stats[i] is None:
                break
            if self.stats[i] < ret:
                ret = self.stats[i]
                index = i
        return self.policies[index], ret
