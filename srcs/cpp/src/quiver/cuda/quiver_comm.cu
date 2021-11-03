#include <c10/cuda/CUDAStream.h>
#include <nccl.h>
#include <pybind11/numpy.h>
#include <string>
#include <torch/extension.h>

namespace quiver
{
py::bytes create_nccl_id()
{
    ncclUniqueId Id;
    ncclGetUniqueId(&Id);
    std::string temp(reinterpret_cast<const char *>(Id.internal),
                     sizeof(Id.internal));
    return py::bytes(temp);
}
class NcclComm
{
  public:
    NcclComm(int rank, int ws, py::bytes id) : rank(rank), size(ws)
    {
        std::string id_str = id;
        memcpy(nccl_id.internal, id_str.data(), sizeof(nccl_id.internal));
        ncclCommInitRank(&nccl_comm, ws, nccl_id, rank);
    }

    int get_rank() { return rank; }

    int get_size() { return size; }

    int get_device()
    {
        int dev;
        ncclCommCuDevice(nccl_comm, &dev);
        return dev;
    }

    void send(torch::Tensor tensor, int dst)
    {
        auto stream = c10::cuda::getCurrentCUDAStream();
        ncclDataType_t type;
        void *ptr;
        ptr_type(tensor, &ptr, &type);
        ncclSend(ptr, tensor.numel(), type, dst, nccl_comm, stream);
    }

    void recv(torch::Tensor tensor, int src)
    {
        auto stream = c10::cuda::getCurrentCUDAStream();
        ncclDataType_t type;
        void *ptr;
        ptr_type(tensor, &ptr, &type);
        ncclRecv(ptr, tensor.numel(), type, src, nccl_comm, stream);
    }

    void allreduce(torch::Tensor tensor)
    {
        auto stream = c10::cuda::getCurrentCUDAStream();
        ncclDataType_t type;
        void *ptr;
        ptr_type(tensor, &ptr, &type);
        ncclAllReduce(ptr, ptr, tensor.numel(), type, ncclSum, nccl_comm,
                      stream);
    }

  private:
    int rank;
    int size;
    ncclComm_t nccl_comm;
    ncclUniqueId nccl_id;
    void ptr_type(torch::Tensor tensor, void **ptr, ncclDataType_t *type)
    {
        if (tensor.options().dtype() == torch::kFloat16) {
            *type = ncclFloat16;
            *ptr = (void *)tensor.data_ptr<at::Half>();
        }
        if (tensor.options().dtype() == torch::kFloat32) {
            *type = ncclFloat32;
            *ptr = (void *)tensor.data_ptr<float>();
        }
        if (tensor.options().dtype() == torch::kInt64) {
            *type = ncclInt64;
            *ptr = (void *)tensor.data_ptr<int64_t>();
        }
    }
};
}  // namespace quiver

void register_cuda_quiver_comm(pybind11::module &m)
{
    m.def("create_nccl_id", &quiver::create_nccl_id);
    py::class_<quiver::NcclComm>(m, "NcclComm")
        .def(py::init<int, int, py::bytes>())
        .def("rank", &quiver::NcclComm::get_rank)
        .def("size", &quiver::NcclComm::get_size)
        .def("device", &quiver::NcclComm::get_device)
        .def("send", &quiver::NcclComm::send)
        .def("recv", &quiver::NcclComm::recv)
        .def("allreduce", &quiver::NcclComm::allreduce);
}