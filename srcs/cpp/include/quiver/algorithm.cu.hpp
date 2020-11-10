#pragma once
#include <cuda_runtime.h>
#include <thrust/distance.h>

namespace quiver
{
struct kernal_option {
    int blocks_per_grid;
    int threads_per_block;
    int shm_size;
};

kernal_option get_kernal_option()
{
    return {
        .blocks_per_grid = 64,
        .threads_per_block = 1024,
        .shm_size = 0,
    };
}

template <class InputIt, class OutputIt, class UnaryOperation>
__host__ void transform_kern(int size, InputIt first1, OutputIt d_first,
                             UnaryOperation f)
{
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int worker_count = gridDim.x * blockDim.x;
    for (int i = worker_idx; i < size; i += worker_count) {
        *(d_first + i) = f(*(first1 + i));
    }
}

template <class InputIt, class OutputIt, class UnaryOperation>
void async_transform(const cudaStream_t stream, InputIt first1, InputIt last1,
                     OutputIt d_first, UnaryOperation f)
{
    const auto o = get_kernal_option();
    const int size = thrust::distance(first1, last1);
    transform_kern<<<o.blocks_per_grid, o.threads_per_block, o.shm_size,
                     stream>>>(size, first1, d_first, f);
}

template <class InputIt, class UnaryOperation>
__host__ void for_each_kern(int size, InputIt first1, UnaryOperation f)
{
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int worker_count = gridDim.x * blockDim.x;
    for (int i = worker_idx; i < size; i += worker_count) { f(*(first1 + i)); }
}

template <class InputIt, class UnaryFunction>
void asyc_for_each(const cudaStream_t stream, InputIt first, InputIt last,
                   UnaryFunction f)
{
    const auto o = get_kernal_option();
    const int size = thrust::distance(first1, last1);
    for_each_kern<<<o.blocks_per_grid, o.threads_per_block, o.shm_size,
                    stream>>>(size, first1, f);
}
}  // namespace quiver
