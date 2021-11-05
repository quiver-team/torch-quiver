from os import lstat
import torch
import torch_sparse
import time
from scipy.sparse import coo, coo_matrix, csr_matrix
import numpy as np
import quiver


def test_metis():
    indptr = torch.load("/data/papers/ogbn_papers100M/csr/indptr.pt")
    indices = torch.load("/data/papers/ogbn_papers100M/csr/indices.pt")
    data = np.zeros(indices.size(0), dtype=np.int32)
    csr_mat = csr_matrix((data, indices.numpy(), indptr.numpy()),
                         shape=(111059956, 111059956))
    graph = dgl.from_scipy(csr_mat)
    print('create graph')
    del indptr
    del indices
    del data
    del csr_mat
    t0 = time.time()
    new_graph = dgl.metis_partition(graph, 2)
    t1 = time.time()
    print(t1 - t0)
    print(new_graph)


def test_random():
    indptr = torch.load("/data/papers/ogbn_papers100M/csr/indptr_bi.pt")
    indices = torch.load("/data/papers/ogbn_papers100M/csr/indices_bi.pt")
    train_idx = torch.load("/data/papers/ogbn_papers100M/index/train_idx.pt")
    idx_len = train_idx.size(0)
    nodes = indptr.size(0) - 1
    total = torch.zeros((nodes, ), device=0)
    remote = torch.ones((nodes, ), device=0)
    cache_rate = 0.5
    cache_id = torch.randperm(nodes)[:int(cache_rate * nodes)].to(0)
    remote[cache_id] = 0
    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    # csr_topo.feature_order = dataset.new_order
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5],
                                                 0,
                                                 mode="UVA")
    # train_idx = train_idx[torch.randperm(idx_len)]
    train_loader = torch.utils.data.DataLoader(train_idx,
                                               batch_size=1024,
                                               pin_memory=True,
                                               shuffle=True)
    cnt = 0
    for i in range(3):
        for seeds in train_loader:
            n_id, _, _ = quiver_sampler.sample(seeds)
            total[n_id] += remote[n_id]
            cnt += n_id.size(0)
        print(i)
        print(torch.sum(total) / cnt)


def test_hot():
    indptr = torch.load("/data/papers/ogbn_papers100M/csr/indptr_bi.pt")
    indices = torch.load("/data/papers/ogbn_papers100M/csr/indices_bi.pt")
    train_idx = torch.load("/data/papers/ogbn_papers100M/index/train_idx.pt")
    idx_len = train_idx.size(0)
    nodes = indptr.size(0) - 1
    total = torch.zeros((nodes, ), device=0)
    remote = torch.ones((nodes, ), device=0)
    hot_rate = 0.5
    prev = torch.LongTensor(indptr[:-1])
    sub = torch.LongTensor(indptr[1:])
    deg = sub - prev
    _, prev_order = torch.sort(deg, descending=True)
    cache_id = torch.randperm(nodes)[:int(0.5 * nodes)].to(0)
    # remote[cache_id] = 0
    hot_id = prev_order[:int(hot_rate * nodes)].to(0)
    remote[hot_id] = 0
    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    # csr_topo.feature_order = dataset.new_order
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5],
                                                 0,
                                                 mode="UVA")
    # train_idx = train_idx[torch.randperm(idx_len)]
    train_loader = torch.utils.data.DataLoader(train_idx[:idx_len // 2],
                                               batch_size=1024,
                                               pin_memory=True,
                                               shuffle=True)
    cnt = 0
    for i in range(3):
        for seeds in train_loader:
            n_id, _, _ = quiver_sampler.sample(seeds)
            total[n_id] += remote[n_id]
            cnt += n_id.size(0)
        print(i)
        print(torch.sum(total) / cnt)


def test_prob():
    indptr = torch.load("/data/papers/ogbn_papers100M/csr/indptr_bi.pt")
    indices = torch.load("/data/papers/ogbn_papers100M/csr/indices_bi.pt")
    train_idx = torch.load("/data/papers/ogbn_papers100M/index/train_idx.pt")
    idx_len = train_idx.size(0)
    nodes = indptr.size(0) - 1
    train_idx0, train_idx1 = train_idx[:idx_len // 2], train_idx[idx_len // 2:]
    train_idx0 = train_idx0.to(0)
    train_idx1 = train_idx1.to(0)

    total = torch.zeros((nodes, ), device=0)
    remote = torch.ones((nodes, ), device=0)
    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5],
                                                 0,
                                                 mode="UVA")
    t0 = time.time()
    prob0 = quiver_sampler.sample_prob(train_idx0, nodes)
    prob1 = quiver_sampler.sample_prob(train_idx1, nodes)
    nz0 = torch.nonzero(prob0)
    _, prev0 = torch.sort(prob0, descending=True)
    _, prev1 = torch.sort(prob1, descending=True)
    nz1 = torch.nonzero(prob1)
    nz = torch.cat((nz0, nz1))
    unique_nz = torch.unique(nz)
    unique_size = unique_nz.size(0)
    choice = unique_nz[torch.randperm(unique_size)[:unique_size // 2]]
    extra = prev0[:int(1.25 * nodes) - unique_size]

    # _, prev_order = torch.sort(prob_diff[unique_nz], descending=True)
    t1 = time.time()
    print(f'preprocess {t1 - t0}')
    remote[choice] = 0
    remote[extra] = 0
    # train_idx = train_idx[torch.randperm(idx_len)]
    train_loader = torch.utils.data.DataLoader(train_idx[:idx_len // 2],
                                               batch_size=1024,
                                               pin_memory=True,
                                               shuffle=True)
    cnt = 0
    for i in range(3):
        for seeds in train_loader:
            n_id, _, _ = quiver_sampler.sample(seeds)
            total[n_id] += remote[n_id]
            cnt += n_id.size(0)
        print(i)
        print(torch.sum(total) / cnt)


def to_undirected():
    indptr = torch.load("/data/papers/ogbn_papers100M/csr/indptr.pt")
    indices = torch.load("/data/papers/ogbn_papers100M/csr/indices.pt")
    data = np.zeros(indices.size(0), dtype=np.int32)
    csr_mat = csr_matrix((data, indices.numpy(), indptr.numpy()),
                         shape=(111059956, 111059956))
    coo_mat = csr_mat.tocoo()
    row, col = coo_mat.row, coo_mat.col
    new_data = np.zeros(indices.size(0) * 2, dtype=np.int32)
    new_row = np.concatenate((row, col))
    new_col = np.concatenate((col, row))
    return new_data, new_row, new_col


def preprocess():
    new_data, new_row, new_col = to_undirected()
    new_coo_mat = coo_matrix((new_data, (new_row, new_col)))
    new_csr_mat = new_coo_mat.tocsr()
    new_indptr, new_indices = new_csr_mat.indptr.astype(
        np.int64), new_csr_mat.indices.astype(np.int64)
    print(new_indptr.shape)
    torch.save(torch.from_numpy(new_indptr),
               "/data/papers/ogbn_papers100M/csr/indptr_bi.pt")
    torch.save(torch.from_numpy(new_indices),
               "/data/papers/ogbn_papers100M/csr/indices_bi.pt")


# test_metis()
# test_random()
# test_hot()

test_prob()
# preprocess()