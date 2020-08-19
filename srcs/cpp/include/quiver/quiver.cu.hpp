#pragma once
#include <algorithm>

#include <quiver/common.hpp>
#include <quiver/cuda_kernel_option.hpp>
#include <quiver/cuda_pair.cu.hpp>
#include <quiver/cuda_random.cu.hpp>
#include <quiver/quiver.hpp>
#include <quiver/trace.hpp>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace quiver
{
template <typename T>
__device__ int safe_sample(const T *begin, const T *end, const int k,
                           T *outputs, cuda_random_generator &g)
{
    const T cap = end - begin;
    if (cap <= k) {
        for (int i = 0; i < cap; ++i) { outputs[i] = begin[i]; }
        return cap;
    } else {
        std_sample(begin, end, outputs, k, g);
        return k;
    }
}

template <typename T>
__global__ void graph_sampler_kernel(const int n, const int m, const T *row_ptr,
                                     const T *col_idx, const int batch_size,
                                     const T *inputs, const int k, T *outputs,
                                     T *output_counts)
{
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int worker_count = gridDim.x * blockDim.x;
    cuda_random_generator g(worker_idx);
    for (int i = worker_idx; i < batch_size; i += worker_count) {
        const T v = inputs[i];
        const T begin = row_ptr[v];
        const T end = v + 1 < n ? row_ptr[v + 1] : m;
        output_counts[i] =
            safe_sample(col_idx + begin, col_idx + end, k, outputs + i * k, g);
    }
}

class graph_sampler
{
    size_t threads_per_block_;
    size_t max_block_count_;

    static size_t ceil_div(size_t a, size_t b)
    {
        return (a / b) + (a % b ? 1 : 0);
    }

  public:
    graph_sampler(size_t threads_per_block = 16, size_t max_block_count = 1024)
        : threads_per_block_(threads_per_block),
          max_block_count_(max_block_count)
    {
    }

    template <typename T>
    void sample(const size_t n, const size_t m, const T *row_ptr,
                const T *col_idx, const size_t batch_size, const T *inputs,
                const int k, T *outputs, T *output_counts) const
    {
        dim3 threadsPerBlock(std::min(threads_per_block_, batch_size));
        dim3 blocksPerGrid(std::min(max_block_count_,
                                    ceil_div(batch_size, threads_per_block_)));
        graph_sampler_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            n, m, row_ptr, col_idx, batch_size, inputs, k, outputs,
            output_counts);
    }
};

template <typename T>
class get_adj_diff
{
    const T *x;
    const size_t n;
    const size_t tot;

  public:
    get_adj_diff(const T *x, const size_t n, const size_t tot)
        : x(x), n(n), tot(tot)
    {
    }

    __device__ T operator()(T i) const
    {
        const T end = i + 1 < n ? x[i + 1] : tot;
        return end - x[i];
    }
};

template <typename T>
class sample_functor
{
    const T *row_ptr;
    const size_t n;
    const T *col_idx;
    const size_t m;

    T *output;

  public:
    sample_functor(const T *row_ptr, size_t n, const T *col_idx, size_t m,
                   T *output)
        : row_ptr(row_ptr), n(n), col_idx(col_idx), m(m), output(output)
    {
    }

    __device__ void operator()(const thrust::tuple<size_t, T, T, T> &t) const
    {
        cuda_random_generator g(thrust::get<0>(t));
        const T &v = thrust::get<1>(t);
        const T &count = thrust::get<2>(t);
        const T &out_ptr = thrust::get<3>(t);

        const T begin = row_ptr[v];
        const T end = v + 1 < n ? row_ptr[v + 1] : m;

        safe_sample(col_idx + begin, col_idx + end, count, output + out_ptr, g);
    }
};

template <typename T>
class quiver<T, CUDA> : public Quiver
{
    graph_sampler sampler_;

    thrust::device_vector<T> row_ptr_;
    thrust::device_vector<T> col_idx_;

  protected:
    void _launch_sampler(size_t batch_size, thrust::device_ptr<const T> inputs,
                         int k, thrust::device_ptr<T> outputs,
                         thrust::device_ptr<T> output_counts) const
    {
        TRACE(__func__);
        sampler_.sample<T>(row_ptr_.size(), col_idx_.size(),
                           thrust::raw_pointer_cast(row_ptr_.data()),
                           thrust::raw_pointer_cast(col_idx_.data()),
                           batch_size, thrust::raw_pointer_cast(inputs), k,
                           thrust::raw_pointer_cast(outputs),
                           thrust::raw_pointer_cast(output_counts));

        cudaError_t err = cudaDeviceSynchronize();
        check_eq(err, cudaSuccess);
    }

  public:
    quiver(T n, std::vector<std::pair<T, T>> edge_index)
    // : row_ptr_(n), col_idx_(edge_index.size())
    {
        TRACE(__func__);
        {
            TRACE("resize");
            row_ptr_.resize(n);
            col_idx_.resize(edge_index.size());
        }

        using TP = thrust::pair<T, T>;
        using CP = std::pair<T, T>;
        static_assert(sizeof(CP) == sizeof(TP), "");

        {
            TRACE("sort via thrust");
            thrust::device_vector<TP> ei(edge_index.size());
            thrust::copy(reinterpret_cast<const TP *>(edge_index.data()),
                         reinterpret_cast<const TP *>(edge_index.data()) +
                             edge_index.size(),
                         ei.begin());
            thrust::sort(ei.begin(), ei.end());
            thrust::copy(ei.begin(), ei.end(),
                         reinterpret_cast<TP *>(edge_index.data()));
        }
        const auto rc_idx = [&] {
            TRACE("unzip");
            return unzip(edge_index);
        }();
        auto &row_idx = rc_idx.first;
        auto &col_idx = rc_idx.second;
        std::vector<T> row_ptr = [&] {
            TRACE("compress_row_idx");
            return compress_row_idx(n, row_idx);
        }();
        {
            TRACE("thrust::copy");
            thrust::copy(row_ptr.begin(), row_ptr.end(), row_ptr_.begin());
            thrust::copy(col_idx.begin(), col_idx.end(), col_idx_.begin());
        }
        // {
        //     TRACE("thrust::transform");
        //     thrust::transform(ei.begin(), ei.end(), row_ptr_.begin(),
        //                       get1st<T>());
        //     thrust::transform(ei.begin(), ei.end(), col_idx_.begin(),
        //                       get2nd<T>());
        // }
    }

    virtual ~quiver() = default;

    size_t size() const override { return row_ptr_.size(); }

    size_t edge_counts() const override { return col_idx_.size(); }

    device_t device() const override { return CUDA; }

    void degree(thrust::device_ptr<const T> input_begin,
                thrust::device_ptr<const T> input_end,
                thrust::device_ptr<T> output_begin) const
    {
        thrust::transform(
            input_begin, input_end, output_begin,
            get_adj_diff<T>(thrust::raw_pointer_cast(row_ptr_.data()),
                            row_ptr_.size(), col_idx_.size()));
    }

    template <typename Iter>
    void sample(Iter input_begin, Iter input_end, Iter output_ptr_begin,
                Iter output_count_begin,
                thrust::device_ptr<T> output_begin) const
    {
        const size_t len = input_end - input_begin;
        thrust::counting_iterator<size_t> i(0);
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(
            i, input_begin, output_count_begin, output_ptr_begin));
        auto end = thrust::make_zip_iterator(
            thrust::make_tuple(i + len, input_end, output_count_begin + len,
                               output_ptr_begin + len));
        thrust::for_each(
            begin, end,
            sample_functor<T>(
                thrust::raw_pointer_cast(row_ptr_.data()), row_ptr_.size(),
                thrust::raw_pointer_cast(col_idx_.data()), col_idx_.size(),
                thrust::raw_pointer_cast(output_begin)));
    }

    void sample(size_t batch_size, const T *vertices, int k, T *sampled_counts,
                T *results) const
    {
        thrust::device_vector<T> inputs(batch_size);
        {
            TRACE("thrust::copy_h2d(" + std::to_string(batch_size * sizeof(T)) +
                  ")");
            thrust::copy(vertices, vertices + batch_size, inputs.begin());
        }
        thrust::device_vector<T> outputs(batch_size * k);
        thrust::device_vector<T> output_counts(batch_size);
        _launch_sampler(inputs.size(), inputs.data(), k, outputs.data(),
                        output_counts.data());
        {
            TRACE("thrust::copy_d2h(" +
                  std::to_string(batch_size * sizeof(T) * k) + ")");
            thrust::copy(output_counts.begin(), output_counts.end(),
                         sampled_counts);
            thrust::copy(outputs.begin(), outputs.end(), results);
        }
    }

    void sample(const std::vector<int> &vertices, int k) const override
    {
        thrust::device_vector<T> inputs(vertices.size());
        thrust::copy(vertices.begin(), vertices.end(), inputs.begin());
        thrust::device_vector<T> outputs(vertices.size() * k);
        thrust::device_vector<T> output_counts(vertices.size());
        _launch_sampler(inputs.size(), inputs.data(), k, outputs.data(),
                        output_counts.data());
    };
};
}  // namespace quiver
