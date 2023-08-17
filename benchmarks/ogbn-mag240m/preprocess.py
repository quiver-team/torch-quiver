from ogb.lsc import MAG240MDataset
from scipy.sparse import csr
import torch
import quiver
import os
import os.path as osp
from torch_sparse import SparseTensor
import time
import numpy as np
from quiver.partition import partition_without_replication, select_nodes

SCALE = 1
GPU_CACHE_GB = 8
# GPU_IDLE_CACHE_GB = 14
CPU_CACHE_GB = 200
FEATURE_DIM = 768


def get_nonzero(data_path):
    dataset = MAG240MDataset(data_path)

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


# def store_mmap(device, data_dir, raw, processed, selected):
#     CHUNK_SIZE = 100000
#     raw_file = osp.join(data_dir, raw)
#     processed_file = osp.join(data_dir, processed)
#     selected = selected.to(device)
#     raw_array = np.load(raw_file, mmap_mode='r')
#     row, dim = raw_array.shape[0], raw_array.shape[1]
#     feat = torch.zeros((selected.size(0), dim))
#     sorted_ids, prev_order = torch.sort(selected)
#     sorted_ids = sorted_ids.cpu()
#     prev_order = prev_order.cpu()
#     beg = 0
#     end = CHUNK_SIZE
#     cnt = 0
#     while end < selected.size(0):
#         print(cnt)
#         t0 = time.time()
#         chunk_index = sorted_ids[beg:end].numpy()
#         chunk_beg = chunk_index[0]
#         chunk_end = chunk_index[-1]
#         chunk_index -= chunk_beg
#         chunk = raw_array[chunk_beg:chunk_end + 1]
#         print(f'load {time.time() - t0}')
#         torch_chunk = torch.from_numpy(chunk).to(dtype=torch.float32)
#         print(f'trans {time.time() - t0}')
#         feat[prev_order[beg:end]] = torch_chunk[chunk_index]
#         beg = end
#         end = min(selected.size(0), beg + CHUNK_SIZE)
#         cnt += 1
#         t1 = time.time()
#         print(t1 - t0)
#     torch.save(feat, processed_file)


def preprocess(data_path, host, host_size, p2p_group, p2p_size):
    dataset = MAG240MDataset(data_path)
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
                             is_sorted=True)
        adj_t = adj_t.to_symmetric()
        torch.save(adj_t, path)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    if not osp.exists(f'{dataset.dir}/csr'):
        os.mkdir(f'{dataset.dir}/csr')
        adj_t = torch.load(f'{dataset.dir}/paper_to_paper_symmetric.pt')
        indptr, indices, _ = adj_t.csr()
        torch.save(indptr, f'{dataset.dir}/csr/indptr.pt')
        torch.save(indices, f'{dataset.dir}/csr/indices.pt')

    indptr = torch.load(osp.join(data_path,
                                 "mag240m_kddcup2021/csr/indptr.pt"))
    indices = torch.load(
        osp.join(data_path, "mag240m_kddcup2021/csr/indices.pt"))
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

    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [25, 15],
                                                 0,
                                                 mode="UVA")
    host_probs_sum = [None] * host_size
    host_p2p_probs = [None] * host_size
    t0 = time.time()
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
    gpu_size = GPU_CACHE_GB * 1024 * 1024 * 1024 // (FEATURE_DIM * SCALE * 4)
    cpu_size = CPU_CACHE_GB * 1024 * 1024 * 1024 // (FEATURE_DIM * SCALE * 4)
    _, nz = select_nodes(0, host_probs_sum, None)
    res = partition_without_replication(0, host_probs_sum, nz.squeeze())
    global2host = torch.zeros(nodes, dtype=torch.int32, device=0) - 1
    t1 = time.time()
    print(f'prob {t1 - t0}')
    for h in range(host_size):
        global2host[res[h]] = h
    torch.save(global2host.cpu(),
               osp.join(data_path, f'{host_size}h/global2host.pt'))
    t2 = time.time()
    print(f'g2h {t2 - t1}')

    for host in range(host_size):
        choice = res[h]
        local_p2p_probs = host_p2p_probs[host]
        local_probs_sum = host_probs_sum[host]
        local_probs_sum_clone = local_probs_sum.clone()
        local_probs_sum_clone[choice] = -1e6
        choice = res[host].clone()
        local_p2p_probs = host_p2p_probs[host]

        _, local_order = torch.sort(local_probs_sum_clone, descending=True)
        del local_probs_sum_clone
        local_replicate_size = min(
            nz.size(0), cpu_size + gpu_size * p2p_size) - choice.size(0)
        replicate = local_order[:local_replicate_size]
        torch.save(replicate.cpu(),
                   osp.join(data_path, f'{host_size}h/replicate{host}.pt'))
        t3 = time.time()
        print(f'replicate {t3 - t2}')
        local_all = torch.cat([choice, replicate])
        _, local_prev_order = torch.sort(local_probs_sum[local_all],
                                         descending=True)
        local_gpu_order = local_prev_order[:gpu_size * p2p_size]
        local_cpu_order = local_prev_order[gpu_size * p2p_size:]
        local_p2p_probs = [prob[local_all] for prob in local_p2p_probs]
        local_res = partition_without_replication(0, local_p2p_probs,
                                                  local_gpu_order)
        local_cpu_ids = local_all[local_cpu_order]
        local_gpu_orders = torch.cat(local_res)
        local_gpu_ids = [local_all[r] for r in local_res]
        local_orders = torch.cat((local_gpu_orders, local_cpu_order))
        torch.save(local_orders.cpu(),
                   osp.join(data_path, f'{host_size}h/local_order{host}.pt'))
        t4 = time.time()
        print(f'order {t4 - t3}')


# def preprocess_unbalance(host, host_size, p2p_group, p2p_size):
#     dataset = MAG240MDataset("/home/ubuntu/temp/mag")

#     indptr = torch.load("/home/ubuntu/temp/mag/mag240m_kddcup2021/csr/indptr.pt")
#     indices = torch.load("/home/ubuntu/temp/mag/mag240m_kddcup2021/csr/indices.pt")
#     train_idx = torch.from_numpy(dataset.get_idx_split('train')).to(0)
#     idx_len = train_idx.size(0)
#     nodes = indptr.size(0) - 1
#     local_gpus = p2p_group * p2p_size
#     global_gpus = p2p_group * p2p_size * host_size
#     local_p2p_group = p2p_group
#     global_p2p_group = p2p_group * host_size
#     train_idxs = []
#     beg = 0
#     for i in range(global_p2p_group):
#         end = min(idx_len, beg + (idx_len // global_p2p_group))
#         train_idxs.append(train_idx[beg:end])
#         beg = end

#     csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
#     quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [25, 15],
#                                                  0,
#                                                  mode="UVA")
#     host_probs_sum = [None] * host_size
#     host_p2p_probs = [None] * host_size
#     t0 = time.time()
#     for h in range(host_size):
#         host_p2p_probs[h] = []
#         p2p_train_idx = torch.LongTensor([]).to(0)
#         for j in range(p2p_group):
#             gpu_index = h * p2p_group + j
#             gpu_train_idx = train_idxs[gpu_index]
#             p2p_train_idx = torch.cat((p2p_train_idx, gpu_train_idx))
#             p2p_probs = quiver_sampler.sample_prob(gpu_train_idx, nodes)
#             host_p2p_probs[h].append(p2p_probs)
#         probs_sum = quiver_sampler.sample_prob(p2p_train_idx, nodes)
#         host_probs_sum[h] = p2p_probs
#     gpu_size = GPU_CACHE_GB * 1024 * 1024 * 1024 // (768 * SCALE * 4)
#     gpu_idle_size = GPU_IDLE_CACHE_GB * 1024 * 1024 * 1024 // (768 * SCALE * 4)
#     cpu_size = CPU_CACHE_GB * 1024 * 1024 * 1024 // (768 * SCALE * 4)
#     _, nz = select_nodes(0, host_probs_sum, None)
#     res = partition_without_replication(0, host_probs_sum, nz.squeeze())
#     global2host = torch.zeros(nodes, dtype=torch.int32, device=0) - 1
#     t1 = time.time()
#     print(f'prob {t1 - t0}')
#     for h in range(host_size):
#         global2host[res[h]] = h
#     torch.save(global2host.cpu(), f'/home/ubuntu/temp/mag/{host_size}h/global2host.pt')
#     t2 = time.time()
#     print(f'g2h {t2 - t1}')
#     del global2host
#     local_probs_sum = host_probs_sum[host].clone()
#     del host_probs_sum
#     local_p2p_probs = host_p2p_probs[host]
#     del host_p2p_probs
#     choice = res[host].clone()
#     del res

#     local_probs_sum_clone = local_probs_sum.clone()
#     local_probs_sum_clone[choice] = -1e6

#     _, local_order = torch.sort(local_probs_sum_clone, descending=True)
#     del local_probs_sum_clone
#     local_replicate_size = min(
#         nz.size(0), cpu_size + gpu_size + gpu_idle_size * (p2p_size - 1)) - choice.size(0)
#     replicate = local_order[:local_replicate_size]
#     torch.save(replicate.cpu(),
#                f'/home/ubuntu/temp/mag/{host_size}h/replicate{host}.pt')
#     t3 = time.time()
#     print(f'replicate {t3 - t2}')
#     local_all = torch.cat([choice, replicate])
#     _, local_prev_order = torch.sort(local_probs_sum[local_all],
#                                      descending=True)
#     torch.save(local_prev_order.cpu(),
#                f'/home/ubuntu/temp/mag/{host_size}h/local_order{host}.pt')
#     t4 = time.time()
#     print(f'order {t4 - t3}')

def init_feat(host, host_size, p2p_group, p2p_size):
    t = torch.zeros((cpu_size, 768 * SCALE))
    torch.save(t, f'/mnt/data/mag/{host_size}h/cpu_feat{host}.pt')
    del t
    gpu_size = GPU_CACHE_GB * 1024 * 1024 * 1024 // (768 * SCALE * 4)
    cpu_size = CPU_CACHE_GB * 1024 * 1024 * 1024 // (768 * SCALE * 4)
    for gpu in range(p2p_size):
        t = torch.zeros((gpu_size, 768 * SCALE))
        torch.save(t, f'/mnt/data/mag/{host_size}h/gpu_feat{host}_{gpu}.pt')
        del t

preprocess('/data/mag', 0, 1, 2, 4)
# preprocess_unbalance(0, 1, 2, 4)
init_feat(0, 1, 2, 4)
