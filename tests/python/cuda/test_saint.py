from torch_sparse import SparseTensor
import torch
import time
import torch_quiver as qv

i = torch.tensor([[0,0,1,1,2,2,3,3,4,4], [2,3,2,4,0,1,0,4,1,3]])
v = torch.tensor([1,2,3,4,5,6,7,8,9,10])
s = SparseTensor(row=i[0], col=i[1], value=v)
print(s.to_dense())
print(s.row)

i = torch.tensor([[0,0,1,2,2,3,3,4,4,1,1], [2,3,4,0,1,0,4,1,3,2,2]])
v = torch.tensor([1,2,4,5,6,7,8,9,10,3,3])
s = SparseTensor(row=i[0], col=i[1], value=v)
print(s.to_dense())
print(s.E)
#
# node_idx = torch.tensor([0, 1, 2])
# start = time.clock()
# out_cpu, _ = s.saint_subgraph(node_idx)
# end = time.clock()
# print(end-start)
# print(out_cpu.to_dense())

# cuda_device = torch.device('cuda')
# s = s.to(cuda_device)
# node_idx = node_idx.to(cuda_device)
# row, col, value = s.coo()
# rowptr = s.storage.rowptr()
# # subgraph
# start = time.clock()
# data = qv.saint_subgraph(node_idx, rowptr, row, col)
# row, col, edge_index = data
# end = time.clock()
#
# if value is not None:
#     value = value[edge_index]
# print(row.size(0))
# print(col.size(0))
#
# out_cuda = SparseTensor(row=row, rowptr=None, col=col, value=value,
#                    sparse_sizes=(node_idx.size(0), node_idx.size(0)),
#                    is_sorted=True)
# print(end-start,"gpu")
# print(out_cuda.to_dense())