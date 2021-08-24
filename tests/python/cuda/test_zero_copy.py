import torch
import numpy as np 
import scipy.sparse as sp
import torch_quiver as qv
def test_neighbor_full_sampler_cuda():
    graph_size = 10000
    seed_size = 2048    

    graph_adj = np.random.randint(0,2,(graph_size,graph_size)) 
    csr_mat = sp.csr_matrix(graph_adj)
    rowptr = torch.from_numpy(csr_mat.indptr).type(torch.long)
    colptr = torch.from_numpy(csr_mat.indices).type(torch.long)
    edge_ids = torch.LongTensor([1])
    quiver = qv.new_quiver_from_csr_array(rowptr, colptr, edge_ids, 0)

test_neighbor_full_sampler_cuda()
    
    