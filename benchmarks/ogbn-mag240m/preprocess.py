from ogb.lsc import MAG240MDataset
from scipy.sparse import csr
import torch
import quiver
import os
import os.path as osp
from torch._C import device
from torch_sparse import SparseTensor
import time
import numpy as np
from quiver.partition import partition_with_replication, partition_without_replication, select_nodes


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


def store_mmap(device, data_dir, raw, processed, selected):
    CHUNK_SIZE = 100000
    raw_file = osp.join(data_dir, raw)
    processed_file = osp.join(data_dir, processed)
    selected = selected.to(device)
    raw_array = np.load(raw_file, mmap_mode='r')
    row, dim = raw_array.shape[0], raw_array.shape[1]
    feat = torch.zeros((selected.size(0), dim))
    sorted_ids, prev_order = torch.sort(selected)
    sorted_ids = sorted_ids.cpu()
    prev_order = prev_order.cpu()
    beg = 0
    end = CHUNK_SIZE
    cnt = 0
    while end < selected.size(0):
        print(cnt)
        t0 = time.time()
        chunk_index = sorted_ids[beg:end].numpy()
        chunk_beg = chunk_index[0]
        chunk_end = chunk_index[-1]
        chunk_index -= chunk_beg
        chunk = raw_array[chunk_beg:chunk_end + 1]
        print(f'load {time.time() - t0}')
        torch_chunk = torch.from_numpy(chunk).to(dtype=torch.float32)
        print(f'trans {time.time() - t0}')
        feat[prev_order[beg:end]] = torch_chunk[chunk_index]
        beg = end
        end = min(selected.size(0), beg + CHUNK_SIZE)
        cnt += 1
        t1 = time.time()
        print(t1 - t0)
    torch.save(feat, processed_file)


def preprocess(host, host_size, p2p_group, p2p_size):
    GPU_CACHE_GB = 20
    CPU_CACHE_GB = 40
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
    train_idx = torch.from_numpy(dataset.get_idx_split('train')).to(0)
    idx_len = train_idx.size(0)
    nodes = indptr.size(0) - 1
    local_gpus = p2p_group * p2p_size
    global_gpus = p2p_group * p2p_size * host_size
    train_idxs = []
    beg = 0
    for i in range(global_gpus):
        end = min(idx_len, beg + (idx_len // global_gpus))
        train_idxs.append(train_idx[beg:end])
        beg = end
    host_index = train_idxs[local_gpus * host:local_gpus * (host + 1)]

    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [25, 15],
                                                 0,
                                                 mode="UVA")
    host_probs_sum = [None] * host_size
    host_p2p_probs = [None] * host_size
    for h in range(host_size):
        p2p_probs = [None] * p2p_size
        for i in range(p2p_size):
            p2p_train_idx = torch.LongTensor([]).to(0)
            for j in range(p2p_group):
                gpu_index = h * p2p_size * p2p_group + p2p_size * j + i
                gpu_train_idx = train_idxs[gpu_index]
                p2p_train_idx = torch.cat((p2p_train_idx, gpu_train_idx))
            p2p_probs[i] = quiver_sampler.sample_prob(p2p_train_idx, nodes)
        host_p2p_probs[h] = p2p_probs
        probs_sum = torch.zeros_like(p2p_probs[0])
        for i in range(p2p_size):
            probs_sum += p2p_probs[i]
        host_probs_sum[h] = probs_sum
    gpu_size = GPU_CACHE_GB * 1024 * 1024 * 1024 // (768 * 4)
    cpu_size = CPU_CACHE_GB * 1024 * 1024 * 1024 // (768 * 4)
    _, nz = select_nodes(0, host_probs_sum, None)
    res = partition_without_replication(0, host_probs_sum, nz.squeeze())
    global2host = torch.zeros(nodes, dtype=torch.int32, device=0) - 1
    for h in range(host_size):
        global2host[res[h]] = h
    torch.save(global2host.cpu(),
               '/data/mag/mag240m_kddcup2021/processed/paper/global2host.pt')
    local_probs_sum = host_probs_sum[host]
    local_probs_sum[res[host]] = -1e6
    _, local_order = torch.sort(local_probs_sum, descending=True)
    local_replicate_size = min(
        nz.size(0), cpu_size + gpu_size * p2p_size) - res[host].size(0)
    replicate = local_order[:local_replicate_size]
    torch.save(
        replicate.cpu(),
        f'/data/mag/mag240m_kddcup2021/processed/paper/replicate{host}.pt')
    total_range = torch.arange(end=nodes, dtype=torch.int64, device=0)
    local_mask = global2host == host
    local_part = total_range[local_mask]
    local_all = torch.cat([local_part, replicate])
    _, local_prev_order = torch.sort(host_probs_sum[host][local_all],
                                     descending=True)
    local_gpu_order = local_prev_order[:gpu_size * p2p_size]
    local_cpu_order = local_prev_order[gpu_size * p2p_size:]
    local_p2p_probs = [prob[local_all] for prob in host_p2p_probs[host]]
    local_res = partition_without_replication(0, local_p2p_probs,
                                              local_gpu_order)
    local_cpu_ids = local_all[local_cpu_order]
    local_gpu_orders = torch.cat(local_res)
    local_gpu_ids = [local_all[r] for r in local_res]
    local_orders = torch.cat((local_gpu_orders, local_cpu_order))
    torch.save(
        local_orders.cpu(),
        f'/data/mag/mag240m_kddcup2021/processed/paper/local_order{host}.pt')
    store_mmap(0, '/data/mag/mag240m_kddcup2021/processed/paper',
               'node_feat.npy', f'cpu_feat{host}.npy', local_cpu_ids)
    for i in range(p2p_size):
        store_mmap(0, '/data/mag/mag240m_kddcup2021/processed/paper',
                   'node_feat.npy', f'gpu_feat{host}{i}.npy', local_gpu_ids[i])


preprocess(0, 2, 1, 2)
