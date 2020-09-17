#include <torch/extension.h>

namespace quiver
{
void show_tensor_info(const torch::Tensor &t);
}  // namespace quiver

void register_cuda_quiver(pybind11::module &m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("show_tensor_info", &quiver::show_tensor_info);

#ifdef HAVE_CUDA
    register_cuda_quiver(m);
#endif
}
