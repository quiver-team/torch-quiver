from ogb.lsc import MAG240MDataset
from scipy.sparse import csr
import torch
import quiver
import os
import os.path as osp
from torch_sparse import SparseTensor
import time
from quiver.partition import partition_with_replication, partition_without_replication


def get_nonzero():
    dataset = MAG240MDataset("/data/mag")

    train_idx = torch.from_numpy(dataset.get_idx_split('train'))

    path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
    adj_t = torch.load(path)
    indptr, indices, _ = adj_t.csr()
    del adj_t

    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [25, 15],
                                                 0,
                                                 mode="UVA")

    prob = quiver_sampler.sample_prob(train_idx, indptr.size(0) - 1)
    nz = torch.nonzero(prob).to('cpu')
    print("nonzero")
    return nz


def preprocess():
    dataset = MAG240MDataset("/data/mag")
    path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
    if not osp.exists(path):
        t = time.perf_counter()
        print('Converting adjacency matrix...', end=' ', flush=True)
        edge_index = dataset.edge_index('paper', 'cites', 'paper')
        edge_index = torch.from_numpy(edge_index)
        adj_t = SparseTensor(row=edge_index[0],
                             col=edge_index[1],
                             sparse_sizes=(dataset.num_papers,
                                           dataset.num_papers),
                             is_sorted=True).csr
        torch.save(adj_t.to_symmetric(), path)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    if not osp.exists(f'{dataset.dir}/csr'):
        os.mkdir(f'{dataset.dir}/csr')
        adj_t = torch.load(f'{dataset.dir}/paper_to_paper_symmetric.pt')
        indptr, indices, _ = adj_t.csr()
        torch.save(indptr, f'{dataset.dir}/csr/indptr.pt')
        torch.save(indices, f'{dataset.dir}/csr/indices.pt')
    indptr = torch.load("/data/mag/mag240m_kddcup2021/csr/indptr.pt")
    indices = torch.load("/data/mag/mag240m_kddcup2021/csr/indices.pt")
    train_idx = torch.from_numpy(dataset.get_idx_split('train'))
    idx_len = train_idx.size(0)
    train_idx0, train_idx1 = train_idx[:idx_len // 2], train_idx[idx_len // 2:]
    idx_len = train_idx.size(0)
    nodes = indptr.size(0) - 1

    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [25, 15],
                                                 0,
                                                 mode="UVA")

    # prob = quiver_sampler.sample_prob(train_idx, nodes)
    # _, prev_order = torch.sort(prob, descending=True)
    # prev_order = prev_order.cpu()
    # torch.save(prev_order, '/data/mag/mag240m_kddcup2021/processed/paper/prev_order.pt')
    prob0 = quiver_sampler.sample_prob(train_idx0, nodes)
    prob1 = quiver_sampler.sample_prob(train_idx1, nodes)
    prob_sum = prob0 + prob1
    _, prev_order = torch.sort(prob_sum, descending=True)
    gpu_size = 20 * 1024 * 1024 * 1024 // (768 * 4)
    cpu_size = 40 * 1024 * 1024 * 1024 // (768 * 4)
    selected = prev_order[:gpu_size * 2]
    res = partition_without_replication(0, [prob0, prob1], selected)
    selected0 = res[0]
    selected1 = res[1]
    prev_order[:gpu_size * 2] = torch.cat((selected0, selected1))
    prev_order = prev_order.cpu()
    torch.save(prev_order, '/data/mag/mag240m_kddcup2021/processed/paper/prev_order2.pt')


preprocess()