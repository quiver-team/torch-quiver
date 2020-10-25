import torch


def test_cuda():
    assert torch.cuda.is_available()
