from scipy.sparse import csr_matrix
import numpy as np
import torch
import torch_quiver as torch_qv


from typing import List


def color_mat(access_book, device_list):
    device_count = access_book.shape[0]

    device2numa = dict.fromkeys(device_list, -1)
    numa2device = {0: [], 1: []}
    current_numa = 0
    for src_device_idx in range(device_count):
        src_device = device_list[src_device_idx]
        if (device2numa[src_device] == -1):
            device2numa[src_device] = current_numa
            numa2device[current_numa].append(src_device)
            current_numa += 1
            for dst_device_idx in range(device_count):
                if (dst_device_idx != src_device_idx
                        and access_book[src_device_idx, dst_device_idx] == 1):
                    dst_device = device_list[dst_device_idx]
                    device2numa[dst_device] = device2numa[src_device]
                    numa2device[device2numa[src_device]].append(dst_device)

    return device2numa, numa2device


class Topo:

    Numa2Device = {}
    Device2Numa = {}

    def __init__(self, device_list: List[int]) -> None:
        access_book = torch.zeros((len(device_list), len(device_list)))
        for src_index, src_device in enumerate(device_list):
            for dst_index, dst_device in enumerate(device_list):
                if torch_qv.can_device_access_peer(src_device, dst_device):
                    access_book[src_index][dst_index] = 1
                    access_book[dst_index][src_index] = 1
        self.Device2Numa, self.Numa2Device = color_mat(access_book,
                                                       device_list)

    def get_numa_node(self, device_id: int):
        return self.Device2Numa[device_id]

    def info(self):
        if len(self.Numa2Device[0]) > 0:
            print(f"Devices {self.Numa2Device[0]} belong to the same numa domain")
        if len(self.Numa2Device[1]) > 0:
            print(f"Devices {self.Numa2Device[1]} belong to the same numa domain")

def get_csr_from_coo(edge_index):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    node_count = max(np.max(src), np.max(dst))
    data = np.zeros(dst.shape, dtype=np.int32)
    csr_mat = csr_matrix(
        (data, (edge_index[0].numpy(), edge_index[1].numpy())))
    return csr_mat

class CSRTopo:
    def __init__(self, edge_index=None, indptr=None, indices=None, eid=None):
        if edge_index is not None:
            csr_mat = get_csr_from_coo(edge_index)
            self.indptr_ = torch.from_numpy(csr_mat.indptr).type(torch.long)
            self.indices_ = torch.from_numpy(csr_mat.indices).type(torch.long)
        elif indptr is not None and indices is not None:
            if isinstance(indptr, torch.Tensor):
                self.indptr_ = indptr.type(torch.long)
                self.indices_ = indices.type(torch.long)
            elif ininstance(indptr, np.ndarray):
                self.indptr_ = torch.from_numpy(indptr).type(torch.long)
                self.indices_ = torch.from_numpy(indices).type(torch.long)
        self.eid_ = eid
        self.feature_order_ = None
    
    @property
    def indptr(self):
        return self.indptr_
    
    @property
    def indices(self):
        return self.indices_
    
    @property
    def eid(self):
        return self.eid_
    
    @property
    def feature_order(self):
        return self.feature_order_
    
    @feature_order.setter
    def feature_order(self, feature_order):
        self.feature_order_ =  feature_order


def reindex_by_config(adj_csr: CSRTopo, graph_feature, gpu_portion):
    degree = adj_csr.indptr[1:] - adj_csr.indptr[:-1]
    node_count = degree.shape[0]
    _, prev_order = torch.sort(degree, descending=True)
    total_range = torch.arange(node_count, dtype=torch.long)
    perm_range = torch.randperm(int(node_count * gpu_portion))

    new_order = torch.zeros_like(prev_order)
    prev_order[:int(node_count * gpu_portion)] = prev_order[perm_range]
    new_order[prev_order] = total_range
    graph_feature = graph_feature[prev_order]
    return graph_feature, new_order

def reindex_feature(graph: CSRTopo, feature, ratio):
    assert isinstance(graph, CSRTopo), "Input graph should be CSRTopo object"
    feature, new_order = reindex_by_config(graph, feature, ratio)
    return feature, new_order
