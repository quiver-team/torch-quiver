#!/usr/bin/env python3
import torch
import torch_quiver as qv
import time


# edges from larger id to smaller id
def gen_non_uniform_adj(n, index):
    return range(index)


def gen_edge_index(n, gen_adj):
    a = []
    b = []
    for i in range(n):
        dst = gen_adj(n, i)
        if dst is not None:
            src = len(dst) * [i]
            a.extend(src)
            b.extend(dst)
    ei = [a, b]
    return n, torch.LongTensor(ei)


def test_quiver():
    n, ei = gen_edge_index(10, gen_non_uniform_adj)
    eid = torch.LongTensor([100 * ei[0][i] + ei[1][i] for i in range(ei.size(1))])
    g = qv.new_quiver_from_edge_index(n, ei, eid)
    neighbor, eid = g.sample_id(torch.LongTensor([5, 6, 7]), 3)
    print(neighbor)
    print(eid)


def test_quiver_standard():
    n, ei = gen_edge_index(2000, gen_non_uniform_adj)
    eid = torch.LongTensor([10000 * ei[0][i] + ei[1][i] for i in range(ei.size(1))])
    g = qv.new_quiver_from_edge_index(n, ei, eid)
    print('prepared')
    t0 = time.time()
    for i in range(1000):
        g.sample_sub(torch.randint(0, 1000, (1000,), dtype=torch.int64), 10)
    print(time.time() - t0)

test_quiver()
test_quiver_standard()
