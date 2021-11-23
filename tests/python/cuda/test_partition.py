from os import lstat
from numpy.lib.function_base import select
import torch
import torch_sparse
import time
from scipy.sparse import coo, coo_matrix, csr_matrix
import numpy as np
import quiver
#import dgl
from quiver.partition import partition_with_replication, partition_without_replication
from quiver.feature import DeviceConfig, Feature
from ogb.lsc import MAG240MDataset
import os.path as osp
import random
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from matplotlib import pyplot as plt
from torch_geometric.datasets import Reddit




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


def bench_read():
    CHUNK_SIZE = 100000
    np_file = '/data/mag/mag240m_kddcup2021/processed/paper/node_feat.npy'
    raw_array = np.load(np_file, mmap_mode='r')
    row, dim = raw_array.shape[0], raw_array.shape[1]
    beg = 0
    end = CHUNK_SIZE
    cnt = 0
    while end < row:
        print(cnt)
        t0 = time.time()
        chunk = raw_array[beg:end]
        torch_chunk = torch.from_numpy(chunk).to(dtype=torch.float32)
        beg = end
        end = min(row, beg + CHUNK_SIZE)
        cnt += 1
        t1 = time.time()
        print(t1 - t0)


def test_prob():
    indptr = torch.load("/data/mag/mag240m_kddcup2021/csr/indptr.pt")
    indices = torch.load("/data/mag/mag240m_kddcup2021/csr/indices.pt")
    dataset = MAG240MDataset("/data/mag")
    train_idx = torch.from_numpy(dataset.get_idx_split('train'))
    idx_len = train_idx.size(0)
    nodes = indptr.size(0) - 1
    train_idx0, train_idx1 = train_idx[:idx_len // 2], train_idx[idx_len // 2:]
    train_idx0 = train_idx0.to(0)
    train_idx1 = train_idx1.to(0)

    total = torch.zeros((nodes, ), device=0)
    remote = torch.ones((nodes, ), device=0)
    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [25, 15],
                                                 0,
                                                 mode="UVA")
    t0 = time.time()
    # prob = quiver_sampler.sample_prob(train_idx, nodes)
    # prob0 = quiver_sampler.sample_prob(train_idx0, nodes)
    # prob1 = quiver_sampler.sample_prob(train_idx1, nodes)
    # prob_sum = prob0 + prob1
    # _, prev_order = torch.sort(prob_sum, descending=True)
    # gpu_size = 20 * 1024 * 1024 * 1024 // (768 * 4)
    # cpu_size = 40 * 1024 * 1024 * 1024 // (768 * 4)
    # selected = prev_order[:gpu_size * 2]
    # res = partition_without_replication(0, [prob0, prob1], selected)
    # selected0 = res[0]
    # selected1 = res[1]
    # prev_order[:gpu_size * 2] = torch.cat((selected0, selected1))
    # prev_order = prev_order.cpu()
    # torch.save(prev_order, '/data/mag/mag240m_kddcup2021/processed/paper/prev_order2.pt')
    # store_mmap(0, '/data/mag/mag240m_kddcup2021/processed/paper',
    #            'node_feat.npy', 'cpu_feat2.npy', selected)
    # choice = prev_order[:gpu_size + cpu_size]
    exit(0)
    cpu_part = osp.join('/data1', 'cpu_feat.npy')
    gpu_part = osp.join('/data1', 'gpu_feat.npy')
    t1 = time.time()
    print(f'prob {t1 - t0}')
    # _, prev_order = torch.sort(prob_diff[unique_nz], descending=True)
    remote[choice] = 0
    # train_idx = train_idx[torch.randperm(idx_len)]
    train_loader = torch.utils.data.DataLoader(train_idx,
                                               batch_size=1024,
                                               pin_memory=True,
                                               shuffle=True)
    feat = Feature(0, [0], 0, 'device_replicate')
    device_config = DeviceConfig([gpu_part], cpu_part)
    feat.from_mmap(dataset.paper_feat, device_config)
    print(f'from')
    disk_map = torch.zeros(nodes, device=0, dtype=torch.int64) - 1
    mem_range = torch.arange(end=cpu_size + gpu_size,
                             device=0,
                             dtype=torch.int64)
    disk_map[prev_order[:gpu_size + cpu_size]] = mem_range
    feat.set_mmap_file(osp.join('/data1', 'node_feat.npy'), disk_map)
    cnt = 0
    t2 = time.time()
    print(f'feat {t2 - t1}')
    for i in range(2):
        for seeds in train_loader:
            n_id, _, _ = quiver_sampler.sample(seeds)
            t0 = time.time()
            x = feat[n_id]
            t1 = time.time()
            total[n_id] += remote[n_id]
            cnt += n_id.size(0)
            print(t1 - t0)
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


def test_cdf_on_mag240M():
    indptr = torch.load("/data/mag/mag240m_kddcup2021/csr/indptr.pt")
    indices = torch.load("/data/mag/mag240m_kddcup2021/csr/indices.pt")
    dataset = MAG240MDataset("/data/mag")
    train_idx = torch.from_numpy(dataset.get_idx_split('train'))
    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], 0, mode="UVA")
    for partition_num in range(1, 10, 1):
        idx_len = train_idx.size(0)
        hit_cum = torch.zeros_like(csr_topo.indptr)
        random.shuffle(train_idx)
        real_train_idx = train_idx[:idx_len // partition_num]

        train_loader = torch.utils.data.DataLoader(real_train_idx,
                                                batch_size=1024,
                                                pin_memory=True,
                                                shuffle=True)
        for _ in range(10):
            for seeds in train_loader:
                n_id, _, _ = quiver_sampler.sample(seeds)
                hit_cum[n_id] += 1
        
        sorted_hit_cum, _ = torch.sort(hit_cum, descending=True)
        total_hit = torch.sum(sorted_hit_cum)
        levels = []
        x = []
        total_levels = 20
        for level in range(total_levels):
            total_sum_pos = int(1.0 * level * sorted_hit_cum.shape[0] / total_levels)
            total_sum = torch.sum(sorted_hit_cum[:total_sum_pos])
            levels.append(total_sum / total_hit)
            x.append(total_sum_pos / (csr_topo.indptr.shape[0] - 1))
        print(levels)

        plt.plot(x, levels, label=f"partition_num={partition_num}")
        plt.scatter(x, levels, label=f"partition_num={partition_num}")
    plt.savefig("mag240M_30_cdf.png")




def test_cdf_on_paper100M():
    
    indptr = torch.load("/data/papers/ogbn_papers100M/csr/indptr_bi.pt")
    indices = torch.load("/data/papers/ogbn_papers100M/csr/indices_bi.pt")
    train_idx = torch.load("/data/papers/ogbn_papers100M/index/train_idx.pt")
    idx_len = train_idx.size(0)
    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], 0, mode="UVA")
    for partition_num in range(1, 10, 1):
        idx_len = train_idx.size(0)
        hit_cum = torch.zeros_like(csr_topo.indptr)
        random.shuffle(train_idx)
        real_train_idx = train_idx[:idx_len // partition_num]

        train_loader = torch.utils.data.DataLoader(real_train_idx,
                                                batch_size=1024,
                                                pin_memory=True,
                                                shuffle=True)
        for _ in range(10):
            for seeds in train_loader:
                n_id, _, _ = quiver_sampler.sample(seeds)
                hit_cum[n_id] += 1
        
        sorted_hit_cum, _ = torch.sort(hit_cum, descending=True)
        total_hit = torch.sum(sorted_hit_cum)
        levels = []
        x = []
        total_levels = 20
        for level in range(total_levels):
            total_sum_pos = int(1.0 * level * sorted_hit_cum.shape[0] / total_levels)
            total_sum = torch.sum(sorted_hit_cum[:total_sum_pos])
            levels.append(total_sum / total_hit)
            x.append(total_sum_pos / (csr_topo.indptr.shape[0] - 1))
        print(levels)

        plt.plot(x, levels, label=f"partition_num={partition_num}")
        plt.scatter(x, levels, label=f"partition_num={partition_num}")
    plt.savefig("paper100M_30_cdf.png")
 


def test_cdf_on_products():
    
    root = "/home/dalong/data/products/"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']

    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], 0, mode="UVA")


    for partition_num in range(1, 10, 1):
        idx_len = train_idx.size(0)
        hit_cum = torch.zeros_like(csr_topo.indptr)
        random.shuffle(train_idx)
        real_train_idx = train_idx[:idx_len // partition_num]

        train_loader = torch.utils.data.DataLoader(real_train_idx,
                                                batch_size=1024,
                                                pin_memory=True,
                                                shuffle=True)
        for _ in range(10):
            for seeds in train_loader:
                n_id, _, _ = quiver_sampler.sample(seeds)
                hit_cum[n_id] += 1
        
        sorted_hit_cum, _ = torch.sort(hit_cum, descending=True)
        total_hit = torch.sum(sorted_hit_cum)
        levels = []
        x = []
        for level in range(11):
            total_sum_pos = int(1.0 * level * sorted_hit_cum.shape[0] / 10)
            total_sum = torch.sum(sorted_hit_cum[:total_sum_pos])
            levels.append(total_sum / total_hit)
            x.append(total_sum_pos / (csr_topo.indptr.shape[0] - 1))
        print(levels)

        plt.plot(x, levels, label=f"partition_num={partition_num}")
        plt.scatter(x, levels, label=f"partition_num={partition_num}")
    plt.savefig("products_30_cdf.png")

def test_cdf_on_reddit():

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    dataset = Reddit(path)
    data = dataset[0]
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    csr_topo = quiver.CSRTopo(data.edge_index)


    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [25, 10], 0, mode="UVA")


    for partition_num in range(1, 10, 1):
        idx_len = train_idx.size(0)
        hit_cum = torch.zeros_like(csr_topo.indptr)
        random.shuffle(train_idx)
        real_train_idx = train_idx[:idx_len // partition_num]

        train_loader = torch.utils.data.DataLoader(real_train_idx,
                                                batch_size=1024,
                                                pin_memory=True,
                                                shuffle=True)
        for _ in range(10):
            for seeds in train_loader:
                n_id, _, _ = quiver_sampler.sample(seeds)
                hit_cum[n_id] += 1
        
        sorted_hit_cum, _ = torch.sort(hit_cum, descending=True)
        total_hit = torch.sum(sorted_hit_cum)
        levels = []
        x = []
        for level in range(11):
            total_sum_pos = int(1.0 * level * sorted_hit_cum.shape[0] / 10)
            total_sum = torch.sum(sorted_hit_cum[:total_sum_pos])
            levels.append(total_sum / total_hit)
            x.append(total_sum_pos / (csr_topo.indptr.shape[0] - 1))
        print(levels)

        plt.plot(x, levels, label=f"partition_num={partition_num}")
        plt.scatter(x, levels, label=f"partition_num={partition_num}")
    plt.savefig("reddit_30_cdf.png")


    
# test_metis()
# test_random()
# test_hot()

#test_prob()
#test_cdf_on_products()
#test_cdf_on_reddit()
#test_cdf_on_paper100M()
test_cdf_on_mag240M()
# preprocess()
# bench_read()
