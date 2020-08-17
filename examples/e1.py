#!/usr/bin/env python3
import torch

import torch_quiver as qv

def test_types():
    types = [
        torch.IntTensor,
        torch.LongTensor,
        torch.FloatTensor,
        torch.DoubleTensor,
        torch.HalfTensor,
    ]
    for T in types:
        print('testing %s' % (T))
        x = T((1,2,3))
        qv.show_tensor_info(x)

def test_sparse():
    x = torch.zeros((3, 3))
    qv.show_tensor_info(x)
    # shape: (3, 3), dtype: float
    # shape: (3, 3), dtype: float, size: 9, elem_size: 4
    # demangled_type_info_name: N6caffe28TypeMetaE

test_types()
test_sparse()
