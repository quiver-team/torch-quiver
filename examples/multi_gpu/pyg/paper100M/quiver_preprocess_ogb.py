import torch
from ogb.nodeproppred import PygNodePropPredDataset
import os.path as osp
import os
import quiver

def preprocess_ogb(root, name, save=True):
    """
    preprocess paper100M dataset, we do following steps:
        1. Sort Feature
        2. Save Order
    
    """
    dataset = PygNodePropPredDataset(name, root)
    data = dataset[0]
    csr_topo = quiver.CSRTopo(data.edge_index)
    sorted_feature, sorted_order = quiver.utils.reindex_feature(csr_topo, data.x, 0)
    csr_topo.feature_order = sorted_order
    save_dir = osp.join(root, name, "quiver_preprocess")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, mode=0o777)
    if save:
        torch.save(csr_topo, osp.join(save_dir, f"{name}_csr_topo.pth"))
        torch.save(sorted_feature, osp.join(save_dir, f"{name}_sorted_feature.pth"))
    return dataset, sorted_feature, csr_topo

def load_preprocessed_quiver(root, name):
    """
    load sorted feature and csr_topo
    """
    save_dir = osp.join(root, name, "quiver_preprocess")
    csr_topo_path = osp.join(save_dir, f"{name}_csr_topo.pth")
    sorted_feature_path = osp.join(save_dir, f"{name}_sorted_feature.pth")
    csr_topo: quiver.CSRTopo = torch.load(csr_topo_path)
    sorted_feature: torch.Tensor = torch.load(sorted_feature_path)
    return sorted_feature, csr_topo


def preprocess_paper100M(root, name=""):

    dataset, sorted_feature, csr_topo = preprocess_ogb(root, name)
    data = dataset[0]
    save_dir = osp.join(root, name, "quiver_preprocess", "paper100M")
    saved_dataset = {
        "label": data.y,
        "csr_topo": csr_topo, 
        "sorted_feature": sorted_feature,
        "split_idx": dataset.get_idx_split(),
        "edge_index": data.edge_index
    }
    torch.save(saved_dataset, save_dir)
    
if __name__ == "__main__":
    root = "/home/dalong/data/products"
    name = "ogbn-products"
    preprocess_ogb(root, name)
    sorted_feature, csr_topo = load_preprocessed_quiver(root, name)

    print(sorted_feature.stride())
    print(csr_topo)



    

