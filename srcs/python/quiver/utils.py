from scipy.sparse import csr_matrix
import numpy as np
import torch
import torch_quiver as torch_qv
from typing import List

def find_cliques(adj_mat, clique_res, remaining_nodes, potential_clique, skip_nodes):

    if len(remaining_nodes) == 0 and len(skip_nodes) == 0:
        clique_res.append(potential_clique)
        return 1
    
    found_cliques = 0
    for node in remaining_nodes:

        # Try adding the node to the current potential_clique to see if we can make it work.
        new_potential_clique = potential_clique + [node]
        new_remaining_nodes = [n for n in remaining_nodes if adj_mat[node][n] == 1]
        new_skip_list = [n for n in skip_nodes if adj_mat[node][n] == 1]
     
        found_cliques += find_cliques(adj_mat, clique_res, new_remaining_nodes, new_potential_clique, new_skip_list)

        # We're done considering this node.  If there was a way to form a clique with it, we
        # already discovered its maximal clique in the recursive call above.  So, go ahead
        # and remove it from the list of remaining nodes and add it to the skip list.
        remaining_nodes.remove(node)
        skip_nodes.append(node)
    return found_cliques

def color_mat(access_book, device_list):
    device2clique = dict.fromkeys(device_list, -1)
    clique2device = {}
    clique_res = []
    all_nodes = list(range(len(device_list)))
    find_cliques(access_book, clique_res, all_nodes, [], [])
    for index, clique in enumerate(clique_res):
        clique2device[index] = []
        for device_idx in clique:
            clique2device[index].append(device_list[device_idx])
            device2clique[device_list[device_idx]] = index

    return device2clique, clique2device


class Topo:
    """P2P access topology for devices. Normally we use this class to detect the connection topology of GPUs on the machine.
    
    ```python
    >>> p2p_clique_topo = p2pCliqueTopo([0,1])
    >>> print(p2p_clique_topo.info())
    ```

    Args:
        device_list ([int]): device list for detecting p2p access topology
        
    
    """
    def __init__(self, device_list: List[int]) -> None:
        access_book = torch.zeros((len(device_list), len(device_list)))
        for src_index, src_device in enumerate(device_list):
            for dst_index, dst_device in enumerate(device_list):
                if src_index != dst_index and torch_qv.can_device_access_peer(src_device, dst_device):
                    access_book[src_index][dst_index] = 1
                    access_book[dst_index][src_index] = 1
        self.Device2p2pClique, self.p2pClique2Device = color_mat(access_book, device_list)

    def get_clique_id(self, device_id: int):
        """Get clique id for device with device_id 

        Args:
            device_id (int): device id of the device

        Returns:
            int: clique_id of the device
        """
        return self.Device2p2pClique[device_id]

    def info(self):
        """Get string description for p2p access topology, you can call `info()` to check the topology of your GPUs 

        Returns:
            str: p2p access topology for devices in device list
        """
        str = ""
        for clique_idx in self.p2pClique2Device:
            str += f"Devices {self.p2pClique2Device[clique_idx]} support p2p access with each other\n"
        return str
    
    @property
    def p2p_clique(self):
        """get all p2p_cliques constructed from devices in device_list

        Returns:
            Dict : {clique_id:[devices in this clique]}
        """
        return self.p2pClique2Device
    

def get_csr_from_coo(edge_index):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    node_count = max(np.max(src), np.max(dst))
    data = np.zeros(dst.shape, dtype=np.int32)
    csr_mat = csr_matrix(
        (data, (edge_index[0].numpy(), edge_index[1].numpy())))
    return csr_mat

class CSRTopo:
    """Graph topology in CSR format.
    
    ```python
    >>> csr_topo = CSRTopo(edge_index=edge_index)
    >>> csr_topo = CSRTopo(indptr=indptr, indices=indices)
    ```
    
    Args:
        edge_index ([torch.LongTensor], optinal): edge_index tensor for graph topo
        indptr (torch.LongTensor, optinal): indptr for CSR format graph topo
        indices (torch.LongTensor, optinal): indices for CSR format graph topo
    """
    def __init__(self, edge_index=None, indptr=None, indices=None, eid=None):
        if edge_index is not None:
            csr_mat = get_csr_from_coo(edge_index)
            self.indptr_ = torch.from_numpy(csr_mat.indptr).type(torch.long)
            self.indices_ = torch.from_numpy(csr_mat.indices).type(torch.long)
        elif indptr is not None and indices is not None:
            if isinstance(indptr, torch.Tensor):
                self.indptr_ = indptr.type(torch.long)
                self.indices_ = indices.type(torch.long)
            elif isinstance(indptr, np.ndarray):
                self.indptr_ = torch.from_numpy(indptr).type(torch.long)
                self.indices_ = torch.from_numpy(indices).type(torch.long)
        self.eid_ = eid
        self.feature_order_ = None
    
    @property
    def indptr(self):
        """Get indptr

        Returns:
            torch.LongTensor: indptr 
        """
        return self.indptr_
    
    @property
    def indices(self):
        """Get indices

        Returns:
            torch.LongTensor: indices
        """
        return self.indices_
    
    @property
    def eid(self):
        return self.eid_
    
    @property
    def feature_order(self):
        """Get feature order for this graph

        Returns:
            torch.LongTensor: feature order 
        """
        return self.feature_order_
    
    @feature_order.setter
    def feature_order(self, feature_order):
        """Set feature order

        Args:
            feature_order (torch.LongTensor): set feature order
        """
        self.feature_order_ =  feature_order
    
    @property
    def degree(self):
        """Get degree of each node in this graph

        Returns:
            [torch.LongTensor]: degree tensor for each node
        """
        return self.indptr[1:] - self.indptr[:-1]
    
    @property
    def node_count(self):
        """Node count of the graph

        Returns:
            int: node count
        """
        return self.indptr_.shape[0] - 1
    
    @property
    def edge_count(self):
        """Edge count of the graph

        Returns:
            int: edge count
        """
        return self.indices_.shape[0] - 1


def reindex_by_config(adj_csr: CSRTopo, graph_feature, gpu_portion):
   
    node_count = adj_csr.indptr.shape[0] - 1
    total_range = torch.arange(node_count, dtype=torch.long)
    perm_range = torch.randperm(int(node_count * gpu_portion))
    # sort and shuffle
    degree = adj_csr.indptr[1:] - adj_csr.indptr[:-1]
    _, prev_order = torch.sort(degree, descending=True)
    new_order = torch.zeros_like(prev_order)
    prev_order[:int(node_count * gpu_portion)] = prev_order[perm_range]
    new_order[prev_order] = total_range
    graph_feature = graph_feature[prev_order]
    return graph_feature, new_order

def reindex_feature(graph: CSRTopo, feature, ratio):
    assert isinstance(graph, CSRTopo), "Input graph should be CSRTopo object"
    feature, new_order = reindex_by_config(graph, feature, ratio)
    return feature, new_order



def init_p2p(device_list: List[int]):
    """Try to enable p2p acess between devices in device_list

    Args:
        device_list (List[int]): device list
    """
    torch_qv.init_p2p(device_list)
