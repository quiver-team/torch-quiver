import torch
import numpy as np 
import scipy.sparse as sp
import torch_quiver as qv
import time
def test_neighbor_full_sampler_cuda():
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
    quiver = qv.new_quiver_from_csr_array(rowptr, colptr, edge_ids, 0)
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
    
    assert torch.allclose(n_id, n_id2)
    
    ##############################
    # CPU Sampling
    ##############################
    quiver = qv.cpu_quiver_from_edge_index(graph_size, edge_index)
    start = time.time()
    n_id3, count3 = quiver.sample_neighbor(0, cuda_seeds, neighbor_size)
    print(f"CPU sampling method consumed {time.time() - start}")
    n_id2_cpu = n_id2.to("cpu")
    assert torch.allclose(n_id2_cpu, n_id3)
    
    
    
    
    
    
test_neighbor_full_sampler_cuda()
    
    