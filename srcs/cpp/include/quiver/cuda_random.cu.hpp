// port several functionalities from <random> for CUDA
#pragma once
#include <curand_kernel.h>

class cuda_random_generator
{
    curandState state_;

  public:
    __device__ cuda_random_generator(uint64_t seed, uint64_t seq = 0,
                                     uint64_t offset = 0)
    {
        curand_init(seed, seq, offset, &state_);
    }

    __device__ uint32_t operator()() { return curand(&state_); }
};

// std::sample for CUDA kernel
template <typename T>
__device__ void std_sample(const T *begin, const T *end, T *outputs, int k,
                           cuda_random_generator &g)
{
    for (int i = 0; i < k; ++i) { outputs[i] = begin[i]; }
    const int n = end - begin;
    for (int i = k; i < n; ++i) {
        // FIXME: make the probability correct
        bool replace = g() % 2;
        if (replace) { outputs[g() % k] = begin[i]; }
    }
}
