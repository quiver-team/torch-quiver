
import torch
from scipy.sparse import csr_matrix
import random
import os.path as osp
import os
import numpy as np
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

def get_csr_from_coo(edge_index):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    node_count = max(np.max(src), np.max(dst))
    data = np.zeros(dst.shape, dtype=np.int32)
    csr_mat = csr_matrix(
        (data, (edge_index[0].numpy(), edge_index[1].numpy())))
    return csr_mat

def reindex_by_config(adj_csr, graph_feature, gpu_portion):
    degree = adj_csr.indptr[1:] - adj_csr.indptr[:-1]
    degree = torch.from_numpy(degree)
    node_count = degree.shape[0]
    _, prev_order = torch.sort(degree, descending=True)
    total_range = torch.arange(node_count, dtype=torch.long)
    perm_range = torch.randperm(int(node_count * gpu_portion))

    new_order = torch.zeros_like(prev_order)
    prev_order[: int(node_count * gpu_portion)] = prev_order[perm_range]
    new_order[prev_order] = total_range
    graph_feature = graph_feature[prev_order]
    return graph_feature, prev_order, new_order, degree[prev_order]


def test_graph_reindex():
    NUM_NODE = 1024
    home = os.getenv('HOME')
    data_dir = osp.join(home, '.pyg')
    root = osp.join(data_dir, 'data', 'products')
    dataset = PygNodePropPredDataset('ogbn-products', root)
    data = dataset[0]
    edge_index = data.edge_index
 
    
    csr_mat = get_csr_from_coo(edge_index)
    node_count = csr_mat.indptr.shape[0] - 1

    print(f"mean degree of graph = {np.mean(csr_mat.indptr[1:] - csr_mat.indptr[:-1])}")

    original_feature = data.x

    new_feature, prev_order, new_order, ordered_degree = reindex_by_config(csr_mat, original_feature, 0.2)

    sampled_node_ids = np.random.randint(0, high=node_count, size=(NUM_NODE))
    mapped_ids = new_order[sampled_node_ids]
    assert torch.equal(original_feature[sampled_node_ids], new_feature[mapped_ids])
    print("Validation SUCCEED!")

    ################
    # Check Stats
    ################
    print(torch.sum(ordered_degree[:int(0.1 * node_count)]) * 1.0 / torch.sum(ordered_degree[int(0.1 * node_count): int(0.2 * node_count)]))
    print(torch.sum(ordered_degree[:int(0.2 * node_count)]) * 1.0  / torch.sum(ordered_degree[int(0.2 * node_count):]))



test_graph_reindex()



    


