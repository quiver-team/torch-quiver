#pragma once
#include <cuda_runtime.h>
#include <thrust/distance.h>
#include <thrust/iterator/zip_iterator.h>

namespace quiver
{
struct kernal_option {
    int blocks_per_grid;
    int threads_per_block;
    int shm_size;
    cudaStream_t stream;

    kernal_option(cudaStream_t stream)
        : blocks_per_grid(128),
          threads_per_block(1024),
          shm_size(0),
          stream(stream)
    {
    }
};

template <class InputIt, class UnaryOperation>
__global__ void for_each_kern(int size, InputIt first, UnaryOperation f)
{
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int worker_count = gridDim.x * blockDim.x;
    for (int i = worker_idx; i < size; i += worker_count) { f(*(first + i)); }
}

template <class InputIt, class UnaryFunction>
void async_for_each(const kernal_option o, InputIt first, InputIt last,
                    UnaryFunction f)
{
    const int size = thrust::distance(first, last);
    for_each_kern<<<o.blocks_per_grid, o.threads_per_block, o.shm_size,
                    o.stream>>>(size, first, f);
}

template <class InputIt, class OutputIt, class UnaryOperation>
void async_transform(const kernal_option o, InputIt first1, InputIt last1,
                     OutputIt d_first, UnaryOperation f)
{
    const int size = thrust::distance(first1, last1);
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(first1, d_first));
    for_each_kern<<<o.blocks_per_grid, o.threads_per_block, o.shm_size,
                    o.stream>>>(size, begin, [f = f] __device__(auto p) {
        thrust::get<1>(p) = f(thrust::get<0>(p));
    });
}
}  // namespace quiver
