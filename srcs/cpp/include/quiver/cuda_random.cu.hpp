// partially implement <random> for CUDA
#pragma once
#include <curand_kernel.h>

constexpr int WARP_SIZE = 32;

/**
 * @brief Perform row-wise sampling on a CSR matrix, and generate a COO matrix,
 * without replacement.
 *
 * @tparam T The ID type used for matrices.
 * @tparam BLOCK_ROWS The number of rows covered by each threadblock.
 * @param rand_seed The random seed to use.
 * @param num_picks The number of non-zeros to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The indptr array of the input CSR.
 * @param in_index The indices array of the input CSR.
 * @param data The data array of the input CSR.
 * @param out_ptr The offset to write each row to in the output COO.
 * @param out_rows The rows of the output COO (output).
 * @param out_cols The columns of the output COO (output).
 * @param out_idxs The data array of the output COO (output).
 */
template <typename T, int BLOCK_WARPS, int TILE_SIZE>
__global__ void CSRRowWiseSampleKernel(
    const uint64_t rand_seed, int num_picks, const int64_t num_rows,
    const T *const in_rows, const T *const in_ptr, const T *const in_index,
    T *const out_ptr, T *const out_count_ptr, T *const out, T *const out_idxs)
{
    // we assign one warp per row
    assert(blockDim.x == WARP_SIZE);
    assert(blockDim.y == BLOCK_WARPS);

    int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
    const int64_t last_row =
        min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

    curandState rng;
    curand_init(rand_seed * gridDim.x + blockIdx.x,
                threadIdx.y * WARP_SIZE + threadIdx.x, 0, &rng);

    while (out_row < last_row) {
        const int64_t row = in_rows[out_row];

        const int64_t in_row_start = in_ptr[row];
        const int64_t deg = in_ptr[row + 1] - in_row_start;

        const int64_t out_row_start = out_ptr[out_row];

        if (deg <= num_picks) {
            // just copy row
            for (int idx = threadIdx.x; idx < deg; idx += WARP_SIZE) {
                const T in_idx = in_row_start + idx;
                out_idxs[out_row_start + idx] = in_index[in_idx];
            }
        } else {
            // generate permutation list via reservoir algorithm
            for (int idx = threadIdx.x; idx < num_picks; idx += WARP_SIZE) {
                out_idxs[out_row_start + idx] = idx;
            }
            __syncwarp();

            for (int idx = num_picks + threadIdx.x; idx < deg;
                 idx += WARP_SIZE) {
                const int num = curand(&rng) % (idx + 1);
                if (num < num_picks) {
                    // use max so as to achieve the replacement order the serial
                    // algorithm would have
                    using Type = unsigned long long int;
                    atomicMax(reinterpret_cast<Type *>(out_idxs +
                                                       out_row_start + num),
                              static_cast<Type>(idx));
                }
            }
            __syncwarp();

            // copy permutation over
            for (int idx = threadIdx.x; idx < num_picks; idx += WARP_SIZE) {
                const T perm_idx = out_idxs[out_row_start + idx] + in_row_start;
                out[out_row_start + idx] = in_index[perm_idx];
            }
        }

        out_row += BLOCK_WARPS;
    }
}

class cuda_base_generator
{
  protected:
    curandState state_;

  public:
    __device__ cuda_base_generator(uint64_t seed, uint64_t seq = 0,
                                   uint64_t offset = 0)
    {
        curand_init(seed, seq, offset, &state_);
    }
};

class cuda_random_generator : public cuda_base_generator
{
  public:
    __device__ cuda_random_generator(uint64_t seed, uint64_t seq = 0,
                                     uint64_t offset = 0)
        : cuda_base_generator(seed, seq, offset)
    {
    }

    __device__ uint32_t operator()() { return curand(&state_); }
};

class cuda_uniform_generator : public cuda_base_generator
{
  public:
    __device__ cuda_uniform_generator(uint64_t seed, uint64_t seq = 0,
                                      uint64_t offset = 0)
        : cuda_base_generator(seed, seq, offset)
    {
    }

    __device__ float operator()() { return curand_uniform(&state_); }
};

// Reservoir sampling
// Reference:
// https://en.wikipedia.org/wiki/Reservoir_sampling
// Random Sampling with a Reservoir
// (http://www.cs.umd.edu/~samir/498/vitter.pdf)
template <typename T>
__device__ void std_sample(const T *begin, const T *end, T *outputs, int k,
                           cuda_random_generator *g)
{
    for (int i = 0; i < k; ++i) { outputs[i] = begin[i]; }
    const int n = end - begin;
    for (int i = k; i < n; ++i) {
        const int j = (*g)() % i;
        if (j < k) { outputs[j] = begin[i]; }
    }
}

template <typename T>
__device__ void std_sample(const T *begin, const T *end, const T *begin_id,
                           T *outputs, T *output_id, int k,
                           cuda_random_generator *g)
{
    for (int i = 0; i < k; ++i) { outputs[i] = begin[i]; }
    for (int i = 0; i < k; ++i) { output_id[i] = begin_id[i]; }
    const int n = end - begin;
    for (int i = k; i < n; ++i) {
        const int j = (*g)() % i;
        if (j < k) {
            outputs[j] = begin[i];
            output_id[j] = begin_id[i];
        }
    }
}

// binary search in exclusive prefix sum
template <typename T, typename W>
__device__ void weight_sample(const T *begin, const T *end,
                              const W *begin_weight, T *outputs, int k,
                              cuda_uniform_generator *g)
{
    const int n = end - begin;
    if (!k || !n) { return; }
    for (int i = 0; i < k; i++) {
        float r = (*g)();
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
    }
}

template <typename T, typename W>
__device__ void weight_sample(const T *begin, const T *end, const T *begin_id,
                              const W *begin_weight, T *outputs, T *output_id,
                              int k, cuda_uniform_generator *g)
{
    const int n = end - begin;
    if (!k || !n) { return; }
    for (int i = 0; i < k; i++) {
        float r = (*g)();
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
__device__ N safe_sample(const T *begin, const T *end, const T *begin_id,
                         const W *begin_weight, const N k, T *outputs,
                         T *output_id, cuda_base_generator *g)
{
    const N cap = end - begin;
    if (begin_weight == nullptr) {
        if (k < cap) {
            if (begin_id != nullptr) {
                std_sample(begin, end, begin_id, outputs, output_id, k,
                           reinterpret_cast<cuda_random_generator *>(g));
            } else {
                std_sample(begin, end, outputs, k,
                           reinterpret_cast<cuda_random_generator *>(g));
            }
            return k;
        } else {
            for (N i = 0; i < cap; ++i) { outputs[i] = begin[i]; }
            if (begin_id != nullptr) {
                for (N i = 0; i < cap; ++i) { output_id[i] = begin_id[i]; }
            }
            return cap;
        }
    } else {
        if (begin_id != nullptr) {
            weight_sample(begin, end, begin_id, begin_weight, outputs,
                          output_id, k,
                          reinterpret_cast<cuda_uniform_generator *>(g));
        } else {
            weight_sample(begin, end, begin_weight, outputs, k,
                          reinterpret_cast<cuda_uniform_generator *>(g));
        }
        return k;
    }
}
