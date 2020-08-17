#!/usr/bin/env python3
import torch
import torch_quiver as qv


def gen_edge_index(n):
    a = []
    b = []
    for i in range(n):
        a.append(i)
        b.append((i + 1) % n)
    ei = [a, b]
    return n, torch.LongTensor(ei)


def test_quiver():
    n, ei = gen_edge_index(10)
    g = qv.new_quiver_from_edge_index(n, ei)
    g.sample_adj(torch.LongTensor([0, 1, 2]), 3)
    assert True


test_quiver()
