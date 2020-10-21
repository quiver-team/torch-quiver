// partially implement <random> for CUDA
#pragma once
#include <curand_kernel.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>

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
    __device__ float gen_uniform_float() { return curand_uniform(&state_); }
};

template<typename T>
class divide_op {
    T to_div;
public:
    divide_op(T d): to_div(d) {

    }
    __device__ T operator() (const T &t) { return t / to_div; }
};

// Reservoir sampling
// Reference:
// https://en.wikipedia.org/wiki/Reservoir_sampling
// Random Sampling with a Reservoir
// (http://www.cs.umd.edu/~samir/498/vitter.pdf)
template <typename T>
__device__ void std_sample(const T *begin, const T *end, const T *begin_id, T *outputs, T *output_id,
                           int k, cuda_random_generator &g)
{
    for (int i = 0; i < k; ++i) { outputs[i] = begin[i]; output_id[i] = begin_id[i]; }
    const int n = end - begin;
    for (int i = k; i < n; ++i) {
        const int j = g() % i;
        if (j < k) { outputs[j] = begin[i]; output_id[j] = begin_id[i]; }
    }
}

template <typename T, typename W>
__device__ void weight_sample(const T *begin, const T *end, const T *begin_id, const W *begin_weight,
                              T *outputs, T *output_id, int k, cuda_random_generator &g)
{
    const int n = end - begin;
    if (!k || !n) {
        return;
    }
    for (int i = 0; i < k; i++) {
        float r = g.gen_uniform_float();
        int lo = 0, hi = n - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (begin_weight[mid] < r) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        outputs[i] = begin[hi];
        output_id[i] = begin_id[hi];
    }
}

// sample at most k elements from [begin, end), returns the sampled count.
template <typename T, typename N, typename W>
__device__ N safe_sample(const T *begin, const T *end, const T *begin_id, const W *begin_weight, const N k,
                         T *outputs, T *output_id, cuda_random_generator &g)
{
    const N cap = end - begin;
    if (begin_weight == nullptr) {
        if (k < cap) {
            std_sample(begin, end, begin_id, outputs, output_id, k, g);
            return k;
        } else {
            for (N i = 0; i < cap; ++i) { outputs[i] = begin[i]; output_id[i] = begin_id[i]; }
            return cap;
        }
    } else {
        weight_sample(begin, end, begin_id, begin_weight, outputs, output_id, k, g);
    }
}
