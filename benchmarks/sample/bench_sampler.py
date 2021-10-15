import torch
import quiver
from ogb.nodeproppred import PygNodePropPredDataset

root = ""
dataset = PygNodePropPredDataset("")
train_loader = torch.utils.data.DataLoader(train_idx, batch_size=1024, pin_memory=True, shuffle=True)