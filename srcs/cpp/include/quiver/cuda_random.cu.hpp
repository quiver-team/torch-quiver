// partially implement <random> for CUDA
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

// Reservoir sampling
// Reference:
// https://en.wikipedia.org/wiki/Reservoir_sampling
// Random Sampling with a Reservoir
// (http://www.cs.umd.edu/~samir/498/vitter.pdf)
template <typename T>
__device__ void std_sample(const T *begin, const T *end, T *outputs, int k,
                           cuda_random_generator &g)
{
    for (int i = 0; i < k; ++i) { outputs[i] = begin[i]; }
    const int n = end - begin;
    for (int i = k; i < n; ++i) {
        const int j = g() % i;
        if (j < k) { outputs[j] = begin[i]; }
    }
}

// sample at most k elements from [begin, end), returns the sampled count.
template <typename T, typename N>
__device__ N safe_sample(const T *begin, const T *end, const N k, T *outputs,
                         cuda_random_generator &g)
{
    const N cap = end - begin;
    if (k < cap) {
        std_sample(begin, end, outputs, k, g);
        return k;
    } else {
        for (N i = 0; i < cap; ++i) { outputs[i] = begin[i]; }
        return cap;
    }
}
