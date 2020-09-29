#pragma once
#include <quiver/common.hpp>
#include <quiver/cuda_pair.cu.hpp>
#include <quiver/cuda_random.cu.hpp>
#include <quiver/quiver.hpp>
#include <quiver/trace.hpp>

#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>

namespace quiver
{
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
void unzip(const thrust::device_vector<thrust::pair<T, T>> &p,
           thrust::device_vector<T> &x, thrust::device_vector<T> &y)
{
    thrust::transform(p.begin(), p.end(), x.begin(), thrust_get<0>());
    thrust::transform(p.begin(), p.end(), y.begin(), thrust_get<1>());
}

template <typename T>
class quiver<T, CUDA>
{
    thrust::device_vector<T> row_ptr_;
    thrust::device_vector<T> col_idx_;

    using TP = thrust::pair<T, T>;
    using CP = std::pair<T, T>;

    static thrust::device_vector<TP>
    to_device(const std::vector<CP> &edge_index)
    {
        static_assert(sizeof(CP) == sizeof(TP), "");
        thrust::device_vector<TP> edge_index_(edge_index.size());
        thrust::copy(reinterpret_cast<const TP *>(edge_index.data()),
                     reinterpret_cast<const TP *>(edge_index.data()) +
                         edge_index.size(),
                     edge_index_.begin());
        return std::move(edge_index_);
    }

  public:
    quiver(T n, const std::vector<CP> &edge_index)
        : quiver(n, to_device(edge_index))
    {
    }

    quiver(T n, thrust::device_vector<TP> edge_index)
        : row_ptr_(n), col_idx_(edge_index.size())
    {
        thrust::sort(edge_index.begin(), edge_index.end());
        thrust::device_vector<T> row_idx_(edge_index.size());
        unzip(edge_index, row_idx_, col_idx_);
        thrust::sequence(row_ptr_.begin(), row_ptr_.end());
        thrust::lower_bound(row_idx_.begin(), row_idx_.end(), row_ptr_.begin(),
                            row_ptr_.end(), row_ptr_.begin());
    }

    virtual ~quiver() = default;

    size_t size() const { return row_ptr_.size(); }

    size_t edge_counts() const { return col_idx_.size(); }

    // device_t device() const   { return CUDA; }

    void degree(const cudaStream_t stream,
                thrust::device_ptr<const T> input_begin,
                thrust::device_ptr<const T> input_end,
                thrust::device_ptr<T> output_begin) const
    {
        thrust::transform(
            thrust::cuda::par.on(stream), input_begin, input_end, output_begin,
            get_adj_diff<T>(thrust::raw_pointer_cast(row_ptr_.data()),
                            row_ptr_.size(), col_idx_.size()));
    }

    template <typename Iter>
    void sample(const cudaStream_t stream, Iter input_begin, Iter input_end,
                Iter output_ptr_begin, Iter output_count_begin,
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
            thrust::cuda::par.on(stream), begin, end,
            sample_functor<T>(
                thrust::raw_pointer_cast(row_ptr_.data()), row_ptr_.size(),
                thrust::raw_pointer_cast(col_idx_.data()), col_idx_.size(),
                thrust::raw_pointer_cast(output_begin)));
    }
};
}  // namespace quiver
