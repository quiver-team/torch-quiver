#pragma once
#include <cuda_runtime.h>
#include <iostream>

namespace quiver
{
class cuda_stream
{
    cudaStream_t stream_;

  public:
    //   TODO: check errors

    cuda_stream() { cudaStreamCreate(&stream_); }

    ~cuda_stream() { cudaStreamDestroy(stream_); }

    operator cudaStream_t() const { return stream_; }
};
}  // namespace quiver
