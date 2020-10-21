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

struct sample_option {
    sample_option(bool w): weighted(w) {

    }
    bool weighted;
};

struct zip_id_functor
{
    template <typename T>
    __device__ void operator()(T tup) {
        thrust::get<0>(tup) = thrust::make_tuple(thrust::get<1>(tup).first, thrust::get<1>(tup).second, thrust::get<2>(tup));
    }
};

struct zip_id_weight_functor
{
    template <typename T>
    __device__ void operator()(T tup) {
        thrust::get<0>(tup) = thrust::make_tuple(thrust::get<1>(tup).first, thrust::get<1>(tup).second, thrust::get<2>(tup), thrust::get<3>(tup));
    }
};

template<typename T, typename W>
class bucket_weight_functor
{
    const W *src_;
    W *dst_;
public:
    bucket_weight_functor(const W *src, W *dst): src_(src), dst_(dst) {

    }
    template<typename P>
    __device__ void operator()(P ptrs) {
        T prev = thrust::get<0>(ptrs);
        T next = thrust::get<1>(ptrs);
        if (prev == next) {
            return;
        }
        W sum = 0;
        for (T temp = prev; temp != next; temp++) {
            dst_[temp] = sum;
            sum += src_[temp];
        }
        for (T temp = prev; temp != next; temp++) {
            dst_[temp] /= sum;
        }
    }
};

template <typename T, typename W>
class sample_functor
{
    const T *row_ptr;
    const size_t n;
    const T *col_idx;
    const T *edge_id;
    const W *edge_weight;
    const size_t m;

    T *output;
    T *output_id;

    bool weighted;

  public:
    sample_functor(const T *row_ptr, size_t n, const T *col_idx, const T *edge_id, const W *edge_weight, size_t m,
                   T *output, T *output_id, bool weighted)
        : row_ptr(row_ptr), n(n), col_idx(col_idx), edge_id(edge_id), edge_weight(edge_weight), m(m),
        output(output), output_id(output_id), weighted(weighted)
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
        const W *begin_weight = weighted ? edge_weight + begin : nullptr;

        safe_sample(col_idx + begin, col_idx + end, edge_id + begin, begin_weight, count,
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

template <typename Ty, typename Tx>
thrust::device_vector<Ty> to_device(const std::vector<Tx> &x)
{
    static_assert(sizeof(Tx) == sizeof(Ty), "");
    thrust::device_vector<Ty> y(x.size());
    thrust::copy(reinterpret_cast<const Ty *>(x.data()),
                reinterpret_cast<const Ty *>(x.data()) + x.size(), y.begin());
    return std::move(y);
}

template <typename T>
void zip_id(thrust::device_vector<thrust::tuple<T, T, T>> &t,
           const thrust::device_vector<thrust::pair<T, T>> &x, const thrust::device_vector<T> &y)
{
    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(t.begin(), x.begin(), y.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(t.end(), x.end(), y.end())),
        zip_id_functor());
}

template <typename T>
void unzip_id(const thrust::device_vector<thrust::tuple<T, T, T>> &t,
           thrust::device_vector<T> &x, thrust::device_vector<T> &y, thrust::device_vector<T> &z)
{
    thrust::transform(t.begin(), t.end(), x.begin(), thrust_get<0>());
    thrust::transform(t.begin(), t.end(), y.begin(), thrust_get<1>());
    thrust::transform(t.begin(), t.end(), z.begin(), thrust_get<2>());
}

template <typename T, typename W>
void zip_id_weight(thrust::device_vector<thrust::tuple<T, T, T, W>> &t,
           const thrust::device_vector<thrust::pair<T, T>> &x, const thrust::device_vector<T> &y, const thrust::device_vector<W> &z)
{
    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(t.begin(), x.begin(), y.begin(), z.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(t.end(), x.end(), y.end(), z.end())),
        zip_id_weight_functor());
}

template <typename T, typename W>
void unzip_id_weight(const thrust::device_vector<thrust::tuple<T, T, T, W>> &t,
           thrust::device_vector<T> &x, thrust::device_vector<T> &y, thrust::device_vector<T> &z, thrust::device_vector<W> &w)
{
    thrust::transform(t.begin(), t.end(), x.begin(), thrust_get<0>());
    thrust::transform(t.begin(), t.end(), y.begin(), thrust_get<1>());
    thrust::transform(t.begin(), t.end(), z.begin(), thrust_get<2>());
    thrust::transform(t.begin(), t.end(), w.begin(), thrust_get<3>());
}

template <typename T, typename W>
void bucket_weight(const thrust::device_vector<T> &prev, const thrust::device_vector<T> &next,
            const thrust::device_vector<W> &src, thrust::device_vector<W> &dst)
{
    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(prev.begin(), next.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(prev.end(), next.end())),
        bucket_weight_functor<T, W>(thrust::raw_pointer_cast(src.data()), thrust::raw_pointer_cast(dst.data())));
}

template <typename T>
class quiver<T, CUDA>
{
    using TP = thrust::pair<T, T>;
    using CP = std::pair<T, T>;
    using W = float;

    thrust::device_vector<T> row_ptr_;
    thrust::device_vector<T> col_idx_;
    thrust::device_vector<T> edge_id_;
    thrust::device_vector<W> edge_weight_;
    thrust::device_vector<W> bucket_edge_weight_;
    
  public:
    quiver(T n, const std::vector<CP> &edge_index, const std::vector<T> &edge_id)
        : quiver(n, to_device<TP>(edge_index), to_device<T>(edge_id))
    {
    }

    quiver(T n, const std::vector<CP> &edge_index, const std::vector<T> &edge_id, const std::vector<W> &edge_weight)
        : quiver(n, to_device<TP>(edge_index), to_device<T>(edge_id), to_device<W>(edge_weight))
    {
    }

    // row_ptr and col_idx make CSR
    quiver(T n, thrust::device_vector<TP> edge_index, thrust::device_vector<T> edge_id)
        : row_ptr_(n), col_idx_(edge_index.size()), edge_id_(std::move(edge_id))
    {
        thrust::device_vector<thrust::tuple<T, T, T>> to_sort(edge_index.size());
        zip_id(to_sort, edge_index, edge_id_);
        thrust::sort(to_sort.begin(), to_sort.end());
        thrust::device_vector<T> row_idx_(edge_index.size());
        unzip_id(to_sort, row_idx_, col_idx_, edge_id_);
        thrust::sequence(row_ptr_.begin(), row_ptr_.end());
        thrust::lower_bound(row_idx_.begin(), row_idx_.end(), row_ptr_.begin(),
                            row_ptr_.end(), row_ptr_.begin());
    }

    quiver(T n, thrust::device_vector<TP> edge_index, thrust::device_vector<T> edge_id, thrust::device_vector<W> edge_weight)
        : row_ptr_(n), col_idx_(edge_index.size()), edge_id_(std::move(edge_id)), edge_weight_(std::move(edge_weight)), 
        bucket_edge_weight_(edge_index.size())
    {
        thrust::device_vector<thrust::tuple<T, T, T, W>> to_sort(edge_index.size());
        zip_id_weight(to_sort, edge_index, edge_id_, edge_weight_);
        thrust::sort(to_sort.begin(), to_sort.end());
        thrust::device_vector<T> row_idx_(edge_index.size());
        thrust::device_vector<T> row_ptr_next(edge_index.size());
        unzip_id_weight(to_sort, row_idx_, col_idx_, edge_id_, edge_weight_);
        thrust::sequence(row_ptr_.begin(), row_ptr_.end());
        thrust::sequence(row_ptr_next.begin(), row_ptr_next.end());
        thrust::lower_bound(row_idx_.begin(), row_idx_.end(), row_ptr_.begin(),
                            row_ptr_.end(), row_ptr_.begin());
        thrust::upper_bound(row_idx_.begin(), row_idx_.end(), row_ptr_next.begin(),
                            row_ptr_next.end(), row_ptr_next.begin());
        bucket_weight(row_ptr_, row_ptr_next, edge_weight_, bucket_edge_weight_);
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
                thrust::device_ptr<T> output_id_begin, sample_option opt) const
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
            sample_functor<T, W>(
                thrust::raw_pointer_cast(row_ptr_.data()), row_ptr_.size(),
                thrust::raw_pointer_cast(col_idx_.data()), 
                thrust::raw_pointer_cast(edge_id_.data()),
                thrust::raw_pointer_cast(bucket_edge_weight_.data()), col_idx_.size(),
                thrust::raw_pointer_cast(output_begin), thrust::raw_pointer_cast(output_id_begin),
                opt.weighted));
    }
};
}  // namespace quiver
