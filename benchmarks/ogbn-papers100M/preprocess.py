import torch
import torch_quiver as qv

import random
import time
import numpy as np
import sys
import torch.multiprocessing as mp
import os.path as osp
from numpy import genfromtxt
from scipy.sparse import csr_matrix
from pathlib import Path

import quiver
from quiver.partition import partition_without_replication, select_nodes

root = '/data'

data_root = f"{root}/ogbn_papers100M/raw/"
label = np.load(osp.join(data_root, "node-label.npz"))
data = np.load(osp.join(data_root, "data.npz"))
path = Path(f'{root}/ogbn_papers100M/feat')
path.mkdir(parents=True)
path = Path(f'{root}/ogbn_papers100M/csr')
path.mkdir(parents=True)
path = Path(f'{root}/ogbn_papers100M/label')
path.mkdir(parents=True)
path = Path(f'{root}/ogbn_papers100M/index')
path.mkdir(parents=True)

SCALE = 1
GPU_CACHE_GB = 4
CPU_CACHE_GB = 18


def get_csr_from_coo(edge_index, reverse=False):
    src = edge_index[0]
    dst = edge_index[1]
    if reverse:
        dst, src = src, dst
    node_count = max(np.max(src), np.max(dst))
    data = np.zeros(dst.shape, dtype=np.int32)
    csr_mat = csr_matrix((data, (src, dst)))
    return csr_mat


def process_topo():
    edge_index = data["edge_index"]
    print("LOG>>> Load Finished")
    num_nodes = data["num_nodes_list"][0]

    print("LOG>>> Begin Process")

    # edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    csr_mat = get_csr_from_coo(edge_index)
    indptr = csr_mat.indptr
    indices = csr_mat.indices
    indptr = torch.from_numpy(indptr).type(torch.long)
    indices = torch.from_numpy(indices).type(torch.long)

    print("LOG>>> Begin Save")

    torch.save(indptr, f"{root}/ogbn_papers100M/csr/indptr.pt")
    torch.save(indices, f"{root}/ogbn_papers100M/csr/indices.pt")

    csr_mat = get_csr_from_coo(edge_index, True)
    indptr_reverse = csr_mat.indptr
    indices_reverse = csr_mat.indices
    indptr_reverse = torch.from_numpy(indptr_reverse).type(torch.long)
    indices_reverse = torch.from_numpy(indices_reverse).type(torch.long)

    torch.save(indptr_reverse,
               f"{root}/ogbn_papers100M/csr/indptr_reverse.pt")
    torch.save(indices_reverse,
               f"{root}/ogbn_papers100M/csr/indices_reverse.pt")


def process_feature():
    print("LOG>>> Load Finished")
    NUM_ELEMENT = data["num_nodes_list"][0]

    nid_feat = data["node_feat"]
    tensor = torch.from_numpy(nid_feat).type(torch.float)
    print("LOG>>> Begin Process")
    torch.save(tensor, f"{root}/ogbn_papers100M/feat/feature.pt")


def process_label():
    print("LOG>>> Load Finished")
    node_label = label["node_label"]
    tensor = torch.from_numpy(node_label).type(torch.long)
    torch.save(tensor, f"{root}/ogbn_papers100M/label/label.pt")


def sort_feature():
    NUM_ELEMENT = 111059956
    indptr = torch.load(f"{root}/ogbn_papers100M/csr/indptr_reverse.pt")
    feature = torch.load(f"{root}/ogbn_papers100M/feat/feature.pt")
    prev = torch.LongTensor(indptr[:-1])
    sub = torch.LongTensor(indptr[1:])
    deg = sub - prev
    sorted_deg, prev_order = torch.sort(deg, descending=True)
    total_num = NUM_ELEMENT
    total_range = torch.arange(total_num, dtype=torch.long)
    feature = feature[prev_order]
    torch.save(feature, f"{root}/ogbn_papers100M/feat/sort_feature.pt")
    torch.save(prev_order, f"{root}/ogbn_papers100M/feat/prev_order.pt")


def process_index():
    data = genfromtxt(f"{root}/ogbn_papers100M/split/time/train.csv",
                      delimiter='\n')
    data = data.astype(np.int_)
    data = torch.from_numpy(data)
    torch.save(data, f"{root}/ogbn_papers100M/index/train_idx.pt")


def preprocess(host, host_size, p2p_group, p2p_size):
    data_dir = osp.join(root, 'ogbn_papers100M')
    indptr_root = osp.join(data_dir, 'csr', 'indptr.pt')
    indices_root = osp.join(data_dir, 'csr', 'indices.pt')
    index_root = osp.join(data_dir, 'index', 'train_idx.pt')
    train_idx = torch.load(index_root).to(0)
    indptr = torch.load(indptr_root)
    indices = torch.load(indices_root)
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
        
    path = Path(f'{root}/ogbn_papers100M/{host_size}h')
    path.mkdir(parents=True)

    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [25, 10],
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
    gpu_size = GPU_CACHE_GB * 1024 * 1024 * 1024 // (128 * SCALE * 4)
    cpu_size = CPU_CACHE_GB * 1024 * 1024 * 1024 // (128 * SCALE * 4)
    _, nz = select_nodes(0, host_probs_sum, None)
    print(f'access node {nz.size(0)}')
    res = partition_without_replication(0, host_probs_sum, nz.squeeze())
    global2host = torch.zeros(nodes, dtype=torch.int32, device=0) - 1
    t1 = time.time()
    print(f'prob {t1 - t0}')
    for h in range(host_size):
        global2host[res[h]] = h
    torch.save(global2host.cpu(), f"{root}/ogbn_papers100M/{host_size}h/global2host.pt")
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
                   f'{root}/ogbn_papers100M/{host_size}h/replicate{host}.pt')
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
                   f'{root}/ogbn_papers100M/{host_size}h/local_order{host}.pt')
        t4 = time.time()
        print(f'order {t4 - t3}')


process_topo()
process_feature()
process_label()
sort_feature()
process_index()

preprocess(0, 3, 1, 2)
