#include <torch/extension.h>

namespace quiver
{
void info(const torch::Tensor &t);
}  // namespace quiver

void register_sparse_matrix(pybind11::module &m);
void register_sparse_matrix_cuda(pybind11::module &m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("show_tensor_info", &quiver::info);
    register_sparse_matrix(m);
#ifdef HAVE_CUDA
    register_sparse_matrix_cuda(m);
#endif
}
