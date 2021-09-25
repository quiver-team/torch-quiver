from ogb.nodeproppred import PygNodePropPredDataset
from scipy.sparse import csr_matrix
import os
import os.path as osp
import numpy as np
import torch
import matplotlib.pyplot as plt


def get_csr_from_coo(edge_index):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    node_count = max(np.max(src), np.max(dst)) + 1
    data = np.zeros(dst.shape, dtype=np.int32)
    csr_mat = csr_matrix(
        (data, (edge_index[0].numpy(), edge_index[1].numpy())))
    return csr_mat


def split(ratio):
    home = os.getenv('HOME')
    data_dir = osp.join(home, '.pyg')
    root = osp.join(data_dir, 'data', 'products')
    dataset = PygNodePropPredDataset('ogbn-products', root)
    data = dataset[0]
    csr = get_csr_from_coo(data.edge_index)
    indptr = csr.indptr
    prev = torch.LongTensor(indptr[:-1])
    sub = torch.LongTensor(indptr[1:])
    deg = sub - prev
    sorted_deg, prev_order = torch.sort(deg, descending=True)
    prev_order.share_memory_()
    total_num = data.x.shape[0]
    total_range = torch.arange(total_num, dtype=torch.long)
    new_order = torch.zeros_like(prev_order)
    new_order[prev_order] = total_range
    new_order.share_memory_()
    last_num = int(ratio[-1] * total_num)
    cpu_tensor = data.x[prev_order[last_num:]].share_memory_()
    index = 0
    res = []
    for i in range(len(ratio) - 1):
        num = int(ratio[i]*total_num)
        gpu_tensor = data.x[prev_order[index:index+num]].share_memory_()
        res.append(gpu_tensor)
    res.append(cpu_tensor)
    return res, prev_order, new_order


# deg = deg.tolist()
# plt.hist(deg, density=False, range=[0, 200], facecolor='g')
# plt.xlabel('Degree')
# plt.ylabel('Probability')
# plt.title('Histogram of products')
# plt.grid(True)
# plt.savefig("hist.png")