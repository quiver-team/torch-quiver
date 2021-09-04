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
        //init_p2p();
    }
    void init_p2p()
    {
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
        //init_p2p();
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
        cudaPointerAttributes attributes;
        cudaPointerGetAttributes(&attributes, (void*) tensor.data_ptr<float>());
        std::cout<< "check device " << attributes.device << " check device pointer" << attributes.devicePointer<<std::endl;
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
static PyObject * shareCuda(PyObject *_self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
  THWStorage *storage = self->cdata;

  if (storage->received_cuda()) {
    AT_ERROR(
        "Attempted to send CUDA tensor received from another process; this is not currently supported. Consider cloning before sending.");
  }

  at::DeviceGuard device_guard(storage->device());
  THPObjectPtr tuple(PyTuple_New(8));
  THPObjectPtr device(THPUtils_packInt32(storage->device().index()));
  THPObjectPtr _handle(Py_None);
  Py_INCREF(Py_None);
  THPObjectPtr size_bytes(THPUtils_packUInt64(storage->nbytes()));
  THPObjectPtr _offset_bytes(THPUtils_packInt32(0));
  THPObjectPtr _ref_counter(Py_None);
  Py_INCREF(Py_None);
  THPObjectPtr _ref_counter_offset(THPUtils_packInt32(0));
  THPObjectPtr _event_handle(Py_None);
  Py_INCREF(Py_None);
  THPObjectPtr _event_sync_required(Py_None);
  Py_INCREF(Py_None);
  if (THWStorage_(data)(LIBRARY_STATE storage)) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    size_t base_size;
    void *base_ptr = c10::cuda::CUDACachingAllocator::getBaseAllocation(THWStorage_(data)(LIBRARY_STATE storage), &base_size);
    ptrdiff_t offset_bytes = (char*)storage->data<scalar_t>() - (char*)base_ptr;

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    cudaIpcMemHandle_t handle;
    THCudaCheck(cudaIpcGetMemHandle(&handle, base_ptr));

    _handle = PyBytes_FromStringAndSize((char *)&handle, CUDA_IPC_HANDLE_SIZE);
    _offset_bytes = PyLong_FromSsize_t((Py_ssize_t)offset_bytes);

    // Put Storage Data behind new ref counting context
    // See Note [CUDA IPC Refcounting implementation explained]
    at::DataPtr sent_data_ptr = torch::GetNewRefCountedSentData(storage->data(), storage->device());
    auto old_data_ptr = storage->set_data_ptr(std::move(sent_data_ptr));
    auto sent_data  =  static_cast<torch::CudaIPCSentData*>(storage->data_ptr().get_context());
    sent_data->set_original_ptr(std::move(old_data_ptr));
    _ref_counter = PyBytes_FromString((sent_data->handle()).c_str());
    _ref_counter_offset = THPUtils_packInt64(sent_data->offset());

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    cudaIpcEventHandle_t ipc_event_handle;

#ifndef __HIP_PLATFORM_HCC__
    if (sent_data->event_sync_required_) {
      THCudaCheck(cudaIpcGetEventHandle(&ipc_event_handle, sent_data->event_));
    }
#else
    // ipc_event_handle unused in storage receiver, we can leave it uninitialized.
#endif

    _event_handle = PyBytes_FromStringAndSize((char *)&ipc_event_handle, CUDA_IPC_HANDLE_SIZE);
    _event_sync_required = PyBool_FromLong(sent_data->event_sync_required_);

  }

  if (!tuple || !device || !_handle || !size_bytes || !_offset_bytes || !_event_handle) {
    return nullptr;
  }
  PyTuple_SET_ITEM(tuple.get(), 0, device.release());
  // cudaIpcMemHandle_t(of basePtr)
  PyTuple_SET_ITEM(tuple.get(), 1, _handle.release());
  // Size(in bytes) of the real storage, note this is not the size of basePtr memory block.
  PyTuple_SET_ITEM(tuple.get(), 2, size_bytes.release());
  // Offset(in bytes) of the real storage in the basePtr memory block.
  // NB: this offset MUST be in bytes instead of numel, since we use (storage_handle, offset)
  //     as key in shared_cache(multiprocessing/reduction.py).
  //     Offset in numel cannot uniquely represent a storage.
  PyTuple_SET_ITEM(tuple.get(), 3, _offset_bytes.release());
  PyTuple_SET_ITEM(tuple.get(), 4, _ref_counter.release());
  PyTuple_SET_ITEM(tuple.get(), 5, _ref_counter_offset.release());
  PyTuple_SET_ITEM(tuple.get(), 6, _event_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 7, _event_sync_required.release());
  return tuple.release();
  END_HANDLE_TH_ERRORS
}
}  // namespace quiver
void register_cuda_quiver_feature(pybind11::module &m)
{
    m.def("share_cda", &quiver::shareCuda);
    py::class_<quiver::ShardTensor>(m, "ShardTensor")
        //.def(py::init<std::vector<torch::Tensor>, int>())
        .def(py::init<int>())
        .def("__getitem__", &quiver::ShardTensor::operator[],
             py::call_guard<py::gil_scoped_release>())
        .def("init_p2p", &quiver::ShardTensor::init_p2p,
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
