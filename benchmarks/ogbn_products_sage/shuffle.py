from ogb.nodeproppred import PygNodePropPredDataset
from scipy.sparse import csr_matrix
import os
import os.path as osp
import numpy as np
import torch


def get_csr_from_coo(edge_index):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    node_count = max(np.max(src), np.max(dst)) + 1
    data = np.zeros(dst.shape, dtype=np.int32)
    csr_mat = csr_matrix(
        (data, (edge_index[0].numpy(), edge_index[1].numpy())))
    return csr_mat


def split(ratio, name, root):
    dataset = PygNodePropPredDataset(name, root)
    data = dataset[0]
    csr = get_csr_from_coo(data.edge_index)
    indptr = csr.indptr
    prev = torch.LongTensor(indptr[:-1])
    sub = torch.LongTensor(indptr[1:])
    deg = sub - prev
    sorted_deg, prev_order = torch.sort(deg, descending=True)
    total_num = data.x.shape[0]
    total_range = torch.arange(total_num, dtype=torch.long)
    if isinstance(ratio, float):
        perm_range = torch.randperm(int(total_num * ratio))
        prev_order[:int(total_num * ratio)] = prev_order[perm_range]
    new_order = torch.zeros_like(prev_order)
    new_order[prev_order] = total_range
    index = 0
    res = []
    if isinstance(ratio, list):
        for i in range(len(ratio) - 1):
            num = int(ratio[i] * total_num)
            gpu_tensor = data.x[prev_order[index:index + num]].share_memory_()
            res.append(gpu_tensor)
            index += num
        cpu_tensor = data.x[prev_order[index:]].clone().share_memory_()
        res.append(cpu_tensor)
    return res, prev_order, new_order


# deg = deg.tolist()
# plt.hist(deg, density=False, range=[0, 200], facecolor='g')
# plt.xlabel('Degree')
# plt.ylabel('Probability')
# plt.title('Histogram of products')
# plt.grid(True)
# plt.savefig("hist.png")
