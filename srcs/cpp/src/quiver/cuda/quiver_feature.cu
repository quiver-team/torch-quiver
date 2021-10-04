#include <ATen/cuda/CUDAContext.h>
#include <pybind11/numpy.h>
#include <quiver/common.hpp>
#include <quiver/quiver.cu.hpp>
#include <quiver/shard_tensor.cu.hpp>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/csrc/utils/python_numbers.h>
//#include <ATen/MapAllocator.h>
#include <atomic>
#include <string>
#include <iostream>

namespace quiver
{
#define CHECK_CPU(x)                                                          \
AT_ASSERTM(!x.device().is_cuda(), #x " must be CPU tensor")

bool can_device_access_peer(int src_device_index, int dst_device_index){
    if (src_device_index == dst_device_index) {
        return true;
    }
    int access_i_j = 0, access_j_i = 0;
    cudaDeviceCanAccessPeer(&access_i_j, src_device_index, dst_device_index);
    cudaDeviceCanAccessPeer(&access_j_i, dst_device_index, src_device_index);
    return (access_i_j == 1) && (access_j_i == 1);
}

class ShardTensorItem
{
  public:
    int device;
    cudaIpcMemHandle_t mem_handle;
    std::vector<int> shape;
    // for now we assume it is all float
    int dtype;
    ShardTensorItem(int device_, cudaIpcMemHandle_t mem_handle_, std::vector<int> shape_):device(device_), mem_handle(mem_handle_), shape(shape_)
    {

    }
    ShardTensorItem(){

    };
    std::tuple<int, py::bytes, std::vector<int>> share_ipc(){
        auto _handle = PyBytes_FromStringAndSize((char *)&mem_handle, CUDA_IPC_HANDLE_SIZE);
        auto bytes_obj = py::reinterpret_steal<py::object>((PyObject*)_handle);
        return std::make_tuple(device, bytes_obj, shape);
    }
    void from_ipc(std::tuple<int, std::string, std::vector<int>> ipc_data){

        device = std::get<0>(ipc_data);
        shape = std::get<2>(ipc_data);
        auto handle = std::get<1>(ipc_data);
        auto ipc_handle = reinterpret_cast<const cudaIpcMemHandle_t*>(handle.c_str());

        mem_handle = *ipc_handle;
    }


};

class ShardTensor
{
  public:
    ShardTensor(int device) : device_(device), inited_(false), device_count_(0)
    {

        offset_list_.push_back(0);
    }

    size_t get_tensor_bytes(torch::Tensor tensor){
        // assume it's float
        int dim = tensor.dim();
        size_t total_bytes = 4;
        for(int index = 0; index < dim; index++){
            total_bytes *= tensor.sizes()[index];
        }
        return total_bytes;
    }

    std::vector<int> get_tensor_shape(torch::Tensor tensor){
        std::vector<int> shape;
        int dim = tensor.dim();
        for(int index = 0; index < dim; index++){
            shape.push_back(tensor.sizes()[index]);
        }
        return shape;
    }

    void append(ShardTensorItem item){
        cudaSetDevice(device_);
        if (!inited_) {
            shape_.resize(item.shape.size());
            shape_[0] = 0;
            auto tensor_sizes = item.shape;
            for (int index = 1; index < shape_.size(); index++) {
                shape_[index] = tensor_sizes[index];
            }
            inited_ = true;
        }
        offset_list_.push_back(offset_list_[offset_list_.size() - 1] + item.shape[0]);

        // Check accessbility
        if(item.device >= 0){
            // TODO
            access_book.push_back(can_device_access_peer(device_, item.device));
        }else{
            access_book.push_back(1);
            //printf("%d <-> CPU support peer access \n", device_);
        }
        // get dev_ptr that can be accessed from this process
        void *ptr = NULL;
        tensor_devices_.push_back(item.device);
        if(!access_book[access_book.size() - 1]){
            cudaSetDevice(item.device);
            cudaIpcOpenMemHandle(&ptr, item.mem_handle, cudaIpcMemLazyEnablePeerAccess);
            cudaSetDevice(device_);
            //printf("WARNING: Tensor from device %d can NOT be accessed in kernel launched on device %d \n", item.device, device_);
        }else{
            cudaIpcOpenMemHandle(&ptr, item.mem_handle, cudaIpcMemLazyEnablePeerAccess);
        }

        //
        dev_ptrs_.push_back((float*)ptr);
        shape_[0] += item.shape[0];
        device_count_ += 1;
        cudaCheckError();

    }

    void append(torch::Tensor &tensor, int target_device)
    {
        CHECK_CPU(tensor);
        // for now, we assume tensor is added ordered
        if (!inited_) {
            shape_.resize(tensor.dim());
            shape_[0] = 0;
            auto tensor_sizes = tensor.sizes();
            for (int index = 1; index < shape_.size(); index++) {
                shape_[index] = tensor_sizes[index];
            }
            inited_ = true;

        }
        tensor_shapes_.push_back(get_tensor_shape(tensor));

        offset_list_.push_back(offset_list_[offset_list_.size() - 1] + tensor.sizes()[0]);

        void *ptr = NULL;
        size_t data_size = get_tensor_bytes(tensor);
        tensor_devices_.push_back(target_device);
        if(target_device >= 0){
            // if target_device >= 0, it means we use p2p
            //printf("LOG >>> Malloc Data On Device %d With %ulld Bytes\n", target_device, data_size);
            cudaSetDevice(target_device);
            cudaMalloc(&ptr, data_size);
            cudaMemcpy(ptr, tensor.data_ptr<float>(), data_size, cudaMemcpyHostToDevice);
            cudaSetDevice(device_);

            // decide access book
            access_book.push_back(can_device_access_peer(device_, target_device));
        }else{
            cudaSetDevice(device_);
            // if target_device < 0, it means we use Zero-Copy
            cudaHostRegister(tensor.data_ptr<float>(), data_size, cudaHostRegisterMapped);
            cudaHostGetDevicePointer(&ptr, (void *)tensor.data_ptr<float>(), 0);
            access_book.push_back(1);
            //printf("%d <-> CPU support peer access \n", device_);
        }

        dev_ptrs_.push_back((float*)ptr);

        shape_[0] += tensor.size(0);
        device_count_ += 1;

    }

    torch::Tensor operator[](torch::Tensor &indices)
    {
        /*
        __global__ void quiver_tensor_gather(const int64_t** dev_ptrs, const
        int64_t* offsets, const int device_count, const int64_t* indices, int
        indice_length, const float* res, const int item_byte_size){
        torch::zeros((100,100),torch::KF32);
        */
        int current_device = 0;
        cudaGetDevice(&current_device);
        auto stream = at::cuda::getCurrentCUDAStream();

        std::vector<int64_t> res_shape(shape_);
        res_shape[0] = indices.numel();
        // decide Tensor
        auto options = torch::TensorOptions()
                           .dtype(at::kFloat)
                           .device(torch::kCUDA, current_device);
        auto res = torch::empty(res_shape, options);
        cudaCheckError();

        // Device Data
        //for(int index = 0; index < offset_list_.size(); index++){
        //    std::cout<<"offset " << offset_list_[index]<<std::endl;
        //    std::cout<<"access_book[index] " << access_book[index]<<std::endl;
        //}

        float **buffers_device;
        int64_t *offset_device;
        int *access_book_device;

        // Copy buffers Device
        cudaMalloc((void ***)&buffers_device, sizeof(float *) * device_count_);
        cudaMemcpy(buffers_device, &dev_ptrs_[0],
                sizeof(float *) * dev_ptrs_.size(), cudaMemcpyHostToDevice);
        cudaCheckError();

        // copy offset
        cudaMalloc((void **)&offset_device,
                sizeof(int64_t) * offset_list_.size());
        cudaMemcpy(offset_device, &offset_list_[0],
                sizeof(int64_t) * offset_list_.size(),
                cudaMemcpyHostToDevice);
        cudaCheckError();

        cudaMalloc((void **)&access_book_device,
                sizeof(int) * access_book.size());
        cudaMemcpy(access_book_device, &access_book[0],
                sizeof(int) * access_book.size(),
                cudaMemcpyHostToDevice);
        cudaCheckError();

        int blockSize = 0;
        int numBlocks = 0;
        cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize,
                                           quiver_tensor_gather);
        // std::cout<<"LOG >>> "<<" numBlocks "<< numBlocks <<" blockSize
        // "<<blockSize<<std::endl;
        int ignore_access_book = 0;
        if(current_device != device_){
            ignore_access_book = 1;
        }
        quiver_tensor_gather<<<numBlocks, blockSize, 0, stream>>>(
            buffers_device, offset_device, offset_list_.size(),
            indices.data_ptr<int64_t>(), indices.numel(), res.data_ptr<float>(),
            stride(0), access_book_device, ignore_access_book);
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
    std::vector<ShardTensorItem> share_ipc(){
        std::vector<ShardTensorItem> res;
        for(int index=0; index < dev_ptrs_.size(); index++){
            if(tensor_devices_[index] >= 0){
                cudaSetDevice(tensor_devices_[index]);
                ShardTensorItem* item = new ShardTensorItem();
                item -> device = tensor_devices_[index];
                item -> shape = tensor_shapes_[index];
                cudaIpcGetMemHandle(&(item -> mem_handle), dev_ptrs_[index]);
                res.push_back(*item);

            }
        }
        return res;
    }

    int device_count() const { return device_count_; }

    void unregister(torch::Tensor& cpu_tensor){

        std::cout<<"begin unregister"<<std::endl;
        cudaHostUnregister((void*)cpu_tensor.data_ptr<float>());
        std::cout<<"end unregister"<<std::endl;

    }


  private:
    std::vector<int64_t> offset_list_;
    std::vector<float *> dev_ptrs_;
    std::vector<int> tensor_devices_;
    std::vector<int> access_book;
    std::vector<std::vector<int>> tensor_shapes_;
    std::vector<int64_t> shape_;
    int numa_broker_device;
    int device_;
    int device_count_;
    bool inited_;


};

void init_p2p(){
    std::cout << "LOG>>> P2P Access Initilization" << std::endl;
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    for (int i = 0; i < numGPUs; i++) {
        cudaSetDevice(i);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        // CUDA IPC is only supported on devices with unified addressing
        if (!prop.unifiedAddressing) {
            printf("Device %d does not support unified addressing, skipping...\n", i);
            continue;
        }
        // This sample requires two processes accessing each device, so we need
        // to ensure exclusive or prohibited mode is not set
        if (prop.computeMode != cudaComputeModeDefault) {
            printf("Device %d is in an unsupported compute mode for this sample\n",
                i);
            continue;
        }

        for (int j = i + 1; j < numGPUs; j++) {
            if (can_device_access_peer(i, j)) {
                printf("Enable P2P Access Between %d <---> %d \n", i, j);
                cudaSetDevice(i);
                cudaDeviceEnablePeerAccess(j, 0);
                cudaCheckError();
                cudaSetDevice(j);
                cudaDeviceEnablePeerAccess(i, 0);
                cudaCheckError();
            }
        }
    }
}
}  // namespace quiver

void register_cuda_quiver_feature(pybind11::module &m)
{
    m.def("init_p2p", &quiver::init_p2p,
            py::call_guard<py::gil_scoped_release>());

    m.def("can_device_access_peer", &quiver::can_device_access_peer,
            py::call_guard<py::gil_scoped_release>());


    py::class_<quiver::ShardTensorItem>(m, "ShardTensorItem")
        .def(py::init<>())
        .def("share_ipc", &quiver::ShardTensorItem::share_ipc)
        .def("from_ipc", &quiver::ShardTensorItem::from_ipc);


    py::class_<quiver::ShardTensor>(m, "ShardTensor")
        //.def(py::init<std::vector<torch::Tensor>, int>())
        .def(py::init<int>())
        .def("__getitem__", &quiver::ShardTensor::operator[],
             py::call_guard<py::gil_scoped_release>())
        .def("unregister", &quiver::ShardTensor::unregister,
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
        .def("append", py::overload_cast<torch::Tensor&, int>(&quiver::ShardTensor::append),
             py::call_guard<py::gil_scoped_release>())
        .def("append", py::overload_cast<quiver::ShardTensorItem>(&quiver::ShardTensor::append),
             py::call_guard<py::gil_scoped_release>())
        .def("share_ipc", &quiver::ShardTensor::share_ipc,
             py::call_guard<py::gil_scoped_release>());
}
