import torch
import torch_quiver as qv


def test_cuda():
    assert torch.cuda.is_available()


def test_construct():
    # TODO: test
    print(qv.new_quiver_from_edge_index)

    # TODO: test
    print(qv.new_quiver_from_edge_index_weight)
    # TODO: test
    print(qv.new_quiver_from_csr_array)
