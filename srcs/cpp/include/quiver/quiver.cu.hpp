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

struct zip_id_functor
{
    template <typename T>
    __device__ void operator()(T tup) {
        thrust::get<0>(tup) = thrust::make_tuple(thrust::get<1>(tup), thrust::get<2>(tup));
    }
};

struct unzip_id_functor
{
    template <typename T>
    __device__ void operator()(T tup) {
        auto zip_ = thrust::get<0>(tup);
        thrust::get<1>(tup) = thrust::get<0>(zip_);
        thrust::get<2>(tup) = thrust::get<1>(zip_);
    }
};

template <typename T>
class sample_functor
{
    const T *row_ptr;
    const size_t n;
    const T *col_idx;
    const T *edge_id;
    const size_t m;

    T *output;
    T *output_id;

  public:
    sample_functor(const T *row_ptr, size_t n, const T *col_idx, const T *edge_id, size_t m,
                   T *output, T *output_id)
        : row_ptr(row_ptr), n(n), col_idx(col_idx), edge_id(edge_id), m(m),
        output(output), output_id(output_id)
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

        safe_sample(col_idx + begin, col_idx + end, edge_id + begin, count,
        output + out_ptr, output_id + out_ptr, g);
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
    thrust::device_vector<T> edge_id_;

    using TP = thrust::pair<T, T>;
    using CP = std::pair<T, T>;

    // copy std::vector<Tx> thrust::device_vecrtor<Ty>
    template <typename Ty, typename Tx>
    thrust::device_vector<Ty> to_device(const std::vector<Tx> &x)
    {
        static_assert(sizeof(Tx) == sizeof(Ty), "");
        thrust::device_vector<Ty> y(x.size());
        thrust::copy(reinterpret_cast<const Ty *>(x.data()),
                    reinterpret_cast<const Ty *>(x.data()) + x.size(), y.begin());
        return std::move(y);
    }
    
  public:
    quiver(T n, const std::vector<CP> &edge_index, const std::vector<T> &edge_id)
        : quiver(n, to_device<TP>(edge_index), to_device<T>(edge_id))
    {
    }

    // row_ptr and col_idx make CSR
    quiver(T n, thrust::device_vector<TP> edge_index, thrust::device_vector<T> edge_id)
        : row_ptr_(n), col_idx_(edge_index.size()), edge_id_(std::move(edge_id))
    {
        thrust::device_vector<thrust::tuple<TP, T>> to_sort(edge_index.size());
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(to_sort.begin(), edge_index.begin(), edge_id_.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(to_sort.end(), edge_index.end(), edge_id_.end())),
            zip_id_functor());
        thrust::sort(to_sort.begin(), to_sort.end());
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(to_sort.begin(), edge_index.begin(), edge_id_.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(to_sort.end(), edge_index.end(), edge_id_.end())),
            unzip_id_functor());
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
                thrust::device_ptr<T> output_begin, 
                thrust::device_ptr<T> output_id_begin) const
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
                thrust::raw_pointer_cast(col_idx_.data()), 
                thrust::raw_pointer_cast(edge_id_.data()), col_idx_.size(),
                thrust::raw_pointer_cast(output_begin), thrust::raw_pointer_cast(output_id_begin)));
    }
};
}  // namespace quiver
