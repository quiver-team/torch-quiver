#pragma once
#include <cuda_runtime.h>

namespace hybrid
{
enum HeteroDevice {
    CPU,
    CUDA,
};
class HeteroWorker
{
    int rank_;
    HeteroDevice device_;
    cudaStream_t stream_;

  public:
    HeteroWorker(int rank, HeteroDevice device, const cudaStream_t &stream_)
        : rank_(rank), device_(device), stream_(stream)
    {
    }
    cudaStream_t get_stream() { return stream_; }
    int get_rank() { return rank_; }
    int get_device() { return device_; }
};

class HeteroAddress
{
    int rank_;
    HeteroDevice device_;
    void *ptr_;

  public:
    HeteroAddress(int rank, HeteroDevice device, void *ptr)
        : rank_(rank), device_(device), ptr_(ptr)
    {
    }
    void *get_ptr() { return ptr_; }
    int get_rank() { return rank_; }
    int get_device() { return device_; }
};
};  // namespace hybrid