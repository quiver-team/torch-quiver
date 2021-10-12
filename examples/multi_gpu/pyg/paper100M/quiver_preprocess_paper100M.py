import torch
from ogb.nodeproppred import PygNodePropPredDataset
import os.path as osp
import quiver

def preprocess_paper100M():
    """
    preprocess paper100M dataset, we do following steps:
        1. Sort Feature
        2. Save Order
    
    """
    dataset = PygNodePropPredDataset("paper100M")
    csr_topo = quiver.CSRTopo(dataset.edge_index)
    sorted_feature, sorted_order = quiver.utils.reindex_feature(csr_topo, dataset.x, 0)
    csr_topo.feature_order = sorted_order
    torch.save(csr_topo, "paper100M_csr_topo.pt")
    torch.save(sorted_feature, "paper100M_sorted_feature.pt")

def load_preprocessed_feature_and_topo(root):
    """
    load sorted feature and csr_topo
    """
    csr_topo_path = osp.join(root, "paper100M_csr_topo.pt")
    sorted_feature_path = osp.join(root, "paper100M_csr_topo.pt")
    csr_topo: quiver.CSRTopo = torch.load(csr_topo_path)
    sorted_feature: torch.Tensor = torch.load(sorted_feature_path)
    return sorted_feature, csr_topo





    

