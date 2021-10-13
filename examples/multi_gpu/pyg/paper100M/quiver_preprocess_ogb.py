import torch
from ogb.nodeproppred import PygNodePropPredDataset
import os.path as osp
import os
import pandas as pd
import numpy as np
import quiver


def preprocess_paper100M(root):
    
    data = np.load("/data/papers/ogbn_papers100M/raw/data.npz")
    print(data.files)
    
    csr_topo = quiver.CSRTopo(torch.from_numpy(data["edge_index"]))
    feature = torch.from_numpy(data["node_feat"])
    label = np.load("/data/papers/ogbn_papers100M/raw/node-label.npz")
    sorted_feature, sorted_order = quiver.utils.reindex_feature(csr_topo, feature, 0)
    csr_topo.feature_order = sorted_order
    save_dir = osp.join(root, "quiver_preprocess")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, mode=0o777)
    save_path = osp.join(save_dir, "paper100M.pth")
    
    train_idx = torch.from_numpy(pd.read_csv(osp.join("/data/papers/ogbn_papers100M/split/time", 'train.csv')).values.T[0]).to(torch.long)
    valid_idx = torch.from_numpy(pd.read_csv(osp.join("/data/papers/ogbn_papers100M/split/time", 'valid.csv')).values.T[0]).to(torch.long)
    test_idx = torch.from_numpy(pd.read_csv(osp.join("/data/papers/ogbn_papers100M/split/time", 'test.csv')).values.T[0]).to(torch.long)
    split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    saved_dataset = {
        "label": torch.from_numpy(label["node_label"]).type(torch.long).squeeze(),
        "csr_topo": csr_topo, 
        "sorted_feature": sorted_feature,
        "split_idx": split_idx,
        "edge_index": torch.from_numpy(data["edge_index"])
    }
    torch.save(saved_dataset, save_path)

if __name__ == "__main__":
    preprocess_paper100M("/data/papers/ogbn_papers100M/")



    

