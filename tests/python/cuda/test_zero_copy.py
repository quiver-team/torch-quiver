import torch
import numpy as np 
import scipy.sparse as sp
import torch_quiver as qv

import time
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from scipy.sparse import csr_matrix
import os
import os.path as osp

from quiver.sage_sampler import GraphSageSampler


def get_csr_from_coo(edge_index):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    node_count = max(np.max(src), np.max(dst))
    data = np.zeros(dst.shape, dtype=np.int32)
    csr_mat = csr_matrix((data, (edge_index[0].numpy(), edge_index[1].numpy())))
    return csr_mat

def test_neighbor_sampler_with_fake_graph():
    print(f"{'*' * 10} TEST WITH FAKE GRAPH {'*' * 10}")
    graph_size = 10000
    seed_size = 2048   
    neighbor_size = 20

    graph_adj = np.random.randint(0,2,(graph_size,graph_size))
    
    ###########################
    # Zero-Copy Sampling
    ############################
    csr_mat = sp.csr_matrix(graph_adj)
    rowptr = torch.from_numpy(csr_mat.indptr).type(torch.long)
    colptr = torch.from_numpy(csr_mat.indices).type(torch.long)
    edge_ids = torch.LongTensor([1])
    quiver = qv.new_quiver_from_csr_array(rowptr, colptr, edge_ids, 0, True, False)
    seeds = np.random.randint(graph_size, size=seed_size)
    seeds = torch.from_numpy(seeds).type(torch.long)
    cuda_seeds = seeds.cuda()
    start = time.time()
    n_id, count = quiver.sample_neighbor(0, cuda_seeds, neighbor_size)
    print(f"Zero-Copy sampling method consumed {time.time() - start}")
    
    ##########################
    # DMA Sampling
    ##########################
    coo_mat = csr_mat.tocoo()
    row = coo_mat.row
    col = coo_mat.col
    row = torch.from_numpy(row).type(torch.long)
    col = torch.from_numpy(col).type(torch.long)
    edge_ids = torch.LongTensor([1])
    edge_index = torch.stack((row, col))
    quiver = qv.new_quiver_from_edge_index(graph_size, edge_index, edge_ids, 0)
    start = time.time()
    n_id2, count2 = quiver.sample_neighbor(0, cuda_seeds, neighbor_size)
    print(f"DMA sampling method consumed {time.time() - start}")
        
    ##############################
    # CPU Sampling
    ##############################
    quiver = qv.cpu_quiver_from_edge_index(graph_size, edge_index)
    start = time.time()
    n_id3, count3 = quiver.sample_neighbor(seeds, neighbor_size)
    print(f"CPU sampling method consumed {time.time() - start}")
    
    
def test_neighbor_sampler_with_real_graph():
    print(f"{'*' * 10} TEST WITH REAL GRAPH {'*' * 10}")
    home = os.getenv('HOME')
    data_dir = osp.join(home, '.pyg')
    root = osp.join(data_dir, 'data', 'products')
    dataset = PygNodePropPredDataset('ogbn-products', root)
    data = dataset[0]
    edge_index = data.edge_index
    seeds_size = 128 * 15 * 10
    neighbor_size = 5
    
    csr_mat = get_csr_from_coo(edge_index)
    print(f"mean degree of graph = {np.mean(csr_mat.indptr[1:] - csr_mat.indptr[:-1])}")
    graph_size = csr_mat.indptr.shape[0] - 1
    seeds = np.arange(graph_size)
    np.random.shuffle(seeds)
    seeds =seeds[:seeds_size]
    
    ###########################
    # Zero-Copy Sampling
    ############################
    rowptr = torch.from_numpy(csr_mat.indptr).type(torch.long)
    colptr = torch.from_numpy(csr_mat.indices).type(torch.long)
    edge_ids = torch.LongTensor([1])
    quiver = qv.new_quiver_from_csr_array(rowptr, colptr, edge_ids, 0, True, False)
    seeds = torch.from_numpy(seeds).type(torch.long)
    cuda_seeds = seeds.cuda()
    start = time.time()
    n_id, count = quiver.sample_neighbor(0, cuda_seeds, neighbor_size)
    print(f"Zero-Copy sampling method consumed {time.time() - start}, sampled res length = {n_id.shape}")
    
    
    ##########################
    # DMA Sampling
    ##########################
    coo_mat = csr_mat.tocoo()
    row = coo_mat.row
    col = coo_mat.col
    row = torch.from_numpy(row).type(torch.long)
    col = torch.from_numpy(col).type(torch.long)
    edge_ids = torch.LongTensor([1])
    quiver = qv.new_quiver_from_edge_index(graph_size, data.edge_index, edge_ids, 0)
    start = time.time()
    n_id2, count2 = quiver.sample_neighbor(0, cuda_seeds, neighbor_size)
    print(f"DMA sampling method consumed {time.time() - start}, sampled res length = {n_id2.shape}")
    
    
    ##############################
    # CPU Sampling
    ##############################
    quiver = qv.cpu_quiver_from_edge_index(graph_size, data.edge_index)
    start = time.time()
    n_id3, count3 = quiver.sample_neighbor(seeds, neighbor_size)
    print(f"CPU sampling method consumed {time.time() - start}, sampled res length = {n_id3.shape}")
    

def test_zero_copy_sampling_gpu_utilization():
    print(f"{'*' * 10} TEST WITH REAL GRAPH {'*' * 10}")
    home = os.getenv('HOME')
    data_dir = osp.join(home, '.pyg')
    root = osp.join(data_dir, 'data', 'products')
    dataset = PygNodePropPredDataset('ogbn-products', root)
    data = dataset[0]
    edge_index = data.edge_index
    seeds_size = 128 * 15 * 10
    neighbor_size = 5
    
    csr_mat = get_csr_from_coo(edge_index)
    print(f"mean degree of graph = {np.mean(csr_mat.indptr[1:] - csr_mat.indptr[:-1])}")
    graph_size = csr_mat.indptr.shape[0] - 1
    seeds = np.arange(graph_size)
    np.random.shuffle(seeds)
    seeds =seeds[:seeds_size]
    
    ###########################
    # Zero-Copy Sampling
    ############################
    rowptr = torch.from_numpy(csr_mat.indptr).type(torch.long)
    colptr = torch.from_numpy(csr_mat.indices).type(torch.long)
    edge_ids = torch.LongTensor([1])
    quiver = qv.new_quiver_from_csr_array(rowptr, colptr, edge_ids, 0, True, False)
    seeds = torch.from_numpy(seeds).type(torch.long)
    cuda_seeds = seeds.cuda()
    start = time.time()
    while True:
        n_id, count = quiver.sample_neighbor(0, cuda_seeds, neighbor_size)
    #print(f"Zero-Copy sampling method consumed {time.time() - start}, sampled res length = {n_id.shape}")
    
    
    
    
#test_neighbor_sampler_with_fake_graph()
test_neighbor_sampler_with_real_graph()
#test_zero_copy_sampling_gpu_utilization()
    
    