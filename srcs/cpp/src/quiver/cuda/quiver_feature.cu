#include <ATen/cuda/CUDAContext.h>
#include <pybind11/numpy.h>
#include <quiver/common.hpp>
#include <quiver/quiver.cu.hpp>
#include <quiver/shard_tensor.cu.hpp>
#include <torch/extension.h>

namespace quiver
{
class ShardTensor
{
  public:
    ShardTensor(std::vector<torch::Tensor> &input_tensor_list, int device)
        : tensor_list_(input_tensor_list), device_(device), inited_(true)
    {
        // init dev_ptrs
        dev_ptrs_.resize(input_tensor_list.size());
        for (int index = 0; index < input_tensor_list.size(); index++) {
            dev_ptrs_[index] = input_tensor_list[index].data_ptr<float>();
        }
        // init offset_list_
        device_count_ = dev_ptrs_.size();
        offset_list_.resize(device_count_);
        offset_list_[0] = 0;
        for (int index = 1; index < device_count_; index++) {
            offset_list_[index] =
                offset_list_[index - 1] + tensor_list_[index - 1].sizes()[0];
        }

        // init shape
        shape_.resize(input_tensor_list[0].dim());
        shape_[0] = 0;
        for (int index = 1; index < shape_.size(); index++) {
            shape_[index] = tensor_list_[0].size(index);
        }
        for (int index = 0; index < tensor_list_.size(); index++) {
            shape_[0] += tensor_list_[index].size(0);
        }
        //
        init_p2p();
    }
    void init_p2p()
    {
        std::cout << "LOG>>> P2P Access Initilization" << std::endl;
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        for (int i = 0; i < numGPUs; i++) {
            cudaSetDevice(i);
            for (int j = i + 1; j < numGPUs; j++) {
                int access = 0;
                cudaDeviceCanAccessPeer(&access, i, j);
                std::cout << "LOG>>> " << i << " " << j << " " << access
                          << std::endl;
                if (access) {
                    cudaSetDevice(i);
                    cudaDeviceEnablePeerAccess(j, 0);
                    cudaCheckError();
                    cudaSetDevice(j);
                    cudaDeviceEnablePeerAccess(i, 0);
                    cudaCheckError();
                }
            }
        }
        cudaSetDevice(device_);

    }
    ShardTensor(int device) : device_(device), inited_(false), device_count_(0)
    {
        init_p2p();
    }
    void append(torch::Tensor &tensor)
    {
        // for now, we assume tensor is added ordered
        if (!inited_) {
            shape_.resize(tensor.dim());
            // std::cout<<"check shape_ size "<<shape_.size()<<std::endl;
            shape_[0] = 0;
            auto tensor_sizes = tensor.sizes();
            for (int index = 1; index < shape_.size(); index++) {
                shape_[index] = tensor_sizes[index];
            }
            inited_ = true;
            offset_list_.push_back(0);
        }
        tensor_list_.push_back(tensor);
        if (device_count_ > 0) {
            offset_list_.push_back(offset_list_[device_count_ - 1] +
                                   tensor.sizes()[0]);
        }
        dev_ptrs_.push_back(tensor.data_ptr<float>());
        shape_[0] += tensor.size(0);
        device_count_ += 1;
    }
    std::tuple<torch::Tensor, long> map(int64_t index)
    {
        for (int i = 0; i < offset_list_.size(); i++) {
            if (index < offset_list_[i]) {
                if (i == 0) { return std::make_tuple(tensor_list_[0], index); }
            } else {
                return std::make_tuple(tensor_list_[i - 1],
                                       index - offset_list_[i - 1]);
            }
        }
        return std::make_tuple(tensor_list_[device_count_ - 1],
                               index - offset_list_[device_count_ - 1]);
    }
    torch::Tensor operator[](torch::Tensor &indices)
    {
        /*
        __global__ void quiver_tensor_gather(const int64_t** dev_ptrs, const
        int64_t* offsets, const int device_count, const int64_t* indices, int
        indice_length, const float* res, const int item_byte_size){
        torch::zeros((100,100),torch::KF32);
        */
        cudaSetDevice(device_);
        auto stream = at::cuda::getCurrentCUDAStream();
        std::vector<int64_t> res_shape(shape_);
        res_shape[0] = indices.numel();
        // decide Tensor
        auto options = torch::TensorOptions()
                           .dtype(tensor_list_[0].dtype())
                           .device(torch::kCUDA, device_);
        auto res = torch::empty(res_shape, options);

        // Copy buffers Device
        float **buffers_device;
        cudaMalloc((void ***)&buffers_device, sizeof(float *) * device_count_);
        cudaMemcpy(buffers_device, &dev_ptrs_[0],
                   sizeof(float *) * dev_ptrs_.size(), cudaMemcpyHostToDevice);
        cudaCheckError();
        // copy offset
        int64_t *offset_device;
        cudaMalloc((void **)&offset_device,
                   sizeof(int64_t) * offset_list_.size());
        cudaMemcpy(offset_device, &offset_list_[0],
                   sizeof(int64_t) * offset_list_.size(),
                   cudaMemcpyHostToDevice);
        cudaCheckError();
        /*
        std::cout << "LOG >>> "
                  << " offset_size " << offset_list_.size() << " Offset Values "
                  << offset_list_[0] << ", " << offset_list_[1] << " stride "
                  << stride(0) << std::endl;
        */
        int blockSize = 0;
        int numBlocks = 0;
        cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize,
                                           quiver_tensor_gather);
        // std::cout<<"LOG >>> "<<" numBlocks "<< numBlocks <<" blockSize
        // "<<blockSize<<std::endl;

        quiver_tensor_gather<<<numBlocks, blockSize, 0, stream>>>(
            buffers_device, offset_device, offset_list_.size(),
            indices.data_ptr<int64_t>(), indices.numel(), res.data_ptr<float>(),
            stride(0));
        cudaCheckError();
        return res;
    }

    std::vector<int64_t> shape() const { return shape_; }

    int device() const { return device_; }

    int size(int dim) const { return shape_[dim]; }

    int64_t stride(int dim) const
    {
        int64_t res = 1;
        for (int index = dim + 1; index < shape_.size(); index++) {
            res *= shape_[index];
        }
        return res;
    }

    int64_t numel() const
    {
        int64_t res = 1;
        for (int index = 0; index < shape_.size(); index++) {
            res *= shape_[index];
        }
        return res;
    }

    int device_count() const { return device_count_; }

  private:
    std::vector<torch::Tensor> tensor_list_;
    std::vector<int64_t> offset_list_;
    std::vector<float *> dev_ptrs_;

    int device_;
    int device_count_;
    std::vector<int64_t> shape_;
    bool inited_;
};
}  // namespace quiver
void register_cuda_quiver_feature(pybind11::module &m)
{
    py::class_<quiver::ShardTensor>(m, "ShardTensor")
        //.def(py::init<std::vector<torch::Tensor>, int>())
        .def(py::init<int>())
        .def("__getitem__", &quiver::ShardTensor::operator[],
             py::call_guard<py::gil_scoped_release>())
        .def("shape", &quiver::ShardTensor::shape,
             py::call_guard<py::gil_scoped_release>())
        .def("numel", &quiver::ShardTensor::numel,
             py::call_guard<py::gil_scoped_release>())
        .def("device", &quiver::ShardTensor::device,
             py::call_guard<py::gil_scoped_release>())
        .def("stride", &quiver::ShardTensor::stride,
             py::call_guard<py::gil_scoped_release>())
        .def("size", &quiver::ShardTensor::size,
             py::call_guard<py::gil_scoped_release>())
        .def("device_count", &quiver::ShardTensor::device_count,
             py::call_guard<py::gil_scoped_release>())
        .def("append", &quiver::ShardTensor::append,
             py::call_guard<py::gil_scoped_release>());
}
