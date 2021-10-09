from scipy.sparse import csr_matrix
import numpy as np
import torch

def reindex_by_config(adj_csr, graph_feature, gpu_portion):
    degree = adj_csr.indptr[1:] - adj_csr.indptr[:-1]
    degree = torch.from_numpy(degree)
    node_count = degree.shape[0]
    _, prev_order = torch.sort(degree, descending=True)
    total_range = torch.arange(node_count, dtype=torch.long)
    perm_range = torch.randperm(int(node_count * gpu_portion))

    new_order = torch.zeros_like(prev_order)
    prev_order[:int(node_count * gpu_portion)] = prev_order[perm_range]
    new_order[prev_order] = total_range
    graph_feature = graph_feature[prev_order]
    return graph_feature, new_order

def get_csr_from_coo(edge_index):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    node_count = max(np.max(src), np.max(dst))
    data = np.zeros(dst.shape, dtype=np.int32)
    csr_mat = csr_matrix(
        (data, (edge_index[0].numpy(), edge_index[1].numpy())))
    return csr_mat

def reindex_feature(graph, feature, ratio):
    if not isinstance(graph, csr_matrix):
        graph = get_csr_from_coo(graph)
    feature, new_order = reindex_by_config(graph, feature, ratio)
    return feature, new_order
