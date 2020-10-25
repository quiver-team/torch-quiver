import torch
import torch_quiver as qv


def test_sparse():
    x = torch.zeros((3, 3))
    qv.show_tensor_info(x)
    assert True
