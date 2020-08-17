#!/usr/bin/env python3
import torch

import torch_quiver as qv

def test_cuda():
    types = [
        torch.cuda.IntTensor,
        torch.cuda.LongTensor,
        torch.cuda.FloatTensor,
        torch.cuda.DoubleTensor,
        torch.cuda.HalfTensor,
    ]
    for T in types:
        print('testing %s' % (T))
        x = T((1,2,3))
        qv.show_tensor_info(x)

def test_sparse():
    edge_index = [[1,2,3], [1,2,3]]
    x = torch.IntTensor(edge_index)
    qv.show_tensor_info(x)
    # shape: (3, 3), dtype: float
    # shape: (3, 3), dtype: float, size: 9, elem_size: 4
    # demangled_type_info_name: N6caffe28TypeMetaE
    spm = qv.new_sparse_matrix_cuda(x)

test_cuda()
test_sparse()
