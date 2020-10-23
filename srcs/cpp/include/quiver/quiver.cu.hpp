#pragma once
#include <quiver/common.hpp>
#include <quiver/cuda_pair.cu.hpp>
#include <quiver/cuda_random.cu.hpp>
#include <quiver/quiver.hpp>
#include <quiver/trace.hpp>
#include <quiver/zip.cu.hpp>

#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <unordered_map>

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
    sample_option(bool w, bool p) : weighted(w), partitioned(p) {}
    bool weighted;
    bool partitioned;
};

template <typename T>
class map_functor
{
    const std::unordered_map<T, T> *map_;
  public:
    map_functor(const std::unordered_map<T, T> *m): map_(m) {}
    T operator()(T t) const
    {
        return map_->find(t)->second;
    }
};

// make edge weight a nomalized prefix sum within each node
template <typename T, typename W>
class bucket_weight_functor
{
    const W *src_;
    W *dst_;

  public:
    bucket_weight_functor(const W *src, W *dst) : src_(src), dst_(dst) {}
    template <typename P>
    __device__ void operator()(P ptrs)
    {
        T prev = thrust::get<0>(ptrs);
        T next = thrust::get<1>(ptrs);
        if (prev == next) { return; }
        W sum = 0;
        for (T temp = prev; temp != next; temp++) {
            dst_[temp] = sum;
            sum += src_[temp];
        }
        for (T temp = prev; temp != next; temp++) { dst_[temp] /= sum; }
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
    sample_functor(const T *row_ptr, size_t n, const T *col_idx,
                   const T *edge_id, const W *edge_weight, size_t m, T *output,
                   T *output_id, bool weighted)
        : row_ptr(row_ptr),
          n(n),
          col_idx(col_idx),
          edge_id(edge_id),
          edge_weight(edge_weight),
          m(m),
          output(output),
          output_id(output_id),
          weighted(weighted)
    {
    }

    __device__ void operator()(const thrust::tuple<size_t, T, T, T> &t) const
    {
        const T &v = thrust::get<1>(t);
        const T &count = thrust::get<2>(t);
        const T &out_ptr = thrust::get<3>(t);

        const T begin = row_ptr[v];
        const T end = v + 1 < n ? row_ptr[v + 1] : m;
        const W *begin_weight = weighted ? edge_weight + begin : nullptr;

        if (!weighted) {
            cuda_random_generator g(thrust::get<0>(t));
            safe_sample(col_idx + begin, col_idx + end, edge_id + begin,
                        begin_weight, count, output + out_ptr,
                        output_id + out_ptr, &g);
        } else {
            cuda_uniform_generator g(thrust::get<0>(t));
            safe_sample(col_idx + begin, col_idx + end, edge_id + begin,
                        begin_weight, count, output + out_ptr,
                        output_id + out_ptr, &g);
        }
    }
};

template <typename T, typename W>
void bucket_weight(const thrust::device_vector<T> &prev,
                   const thrust::device_vector<T> &next,
                   const thrust::device_vector<W> &src,
                   thrust::device_vector<W> &dst)
{
    thrust::for_each(
        thrust::make_zip_iterator(
            thrust::make_tuple(prev.begin(), next.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(prev.end(), next.end())),
        bucket_weight_functor<T, W>(thrust::raw_pointer_cast(src.data()),
                                    thrust::raw_pointer_cast(dst.data())));
}

template <typename T>
void handle_partition(thrust::device_vector<T> &row_idx, thrust::device_vector<T> &local_map_,
                      std::unordered_map<T, T> &m)
{
    local_map_.resize(row_idx.size());
    auto last = thrust::unique_copy(row_idx.begin(), row_idx.end(), local_map_.begin());
    int len = last - local_map_.begin();
    local_map_.resize(len);
    thrust::lower_bound(local_map_.begin(), local_map_.end(), row_idx.begin(), row_idx.end(), row_idx.begin());
    for (int i = 0; i < len; i++) {
        m[local_map_[i]] = i;
    }
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
    thrust::device_vector<T> local_map_;
    std::unordered_map<T, T> remote_map_;
    sample_option opt_;

  public:
    quiver(T n, const std::vector<CP> &edge_index,
           const std::vector<T> &edge_id)
        : quiver(n, to_device<TP>(edge_index), to_device<T>(edge_id))
    {
    }

    quiver(T n, const std::vector<CP> &edge_index,
           const std::vector<T> &edge_id, const std::vector<W> &edge_weight)
        : quiver(n, to_device<TP>(edge_index), to_device<T>(edge_id),
                 to_device<W>(edge_weight))
    {
    }

    // row_ptr and col_idx make CSR
    quiver(T n, thrust::device_vector<TP> edge_index,
           thrust::device_vector<T> edge_id)
        : row_ptr_(n),
          col_idx_(edge_index.size()),
          edge_id_(std::move(edge_id)),
          opt_(false, true)
    {
        thrust::device_vector<thrust::tuple<T, T, T>> to_sort(
            edge_index.size());
        zip(to_sort, edge_index, edge_id_);
        thrust::sort(to_sort.begin(), to_sort.end());
        thrust::device_vector<T> row_idx_(edge_index.size());
        unzip(to_sort, row_idx_, col_idx_, edge_id_);
        handle_partition(row_idx_, local_map_, remote_map_);
        thrust::sequence(row_ptr_.begin(), row_ptr_.end());
        thrust::lower_bound(row_idx_.begin(), row_idx_.end(), row_ptr_.begin(),
                            row_ptr_.end(), row_ptr_.begin());
    }

    quiver(T n, thrust::device_vector<TP> edge_index,
           thrust::device_vector<T> edge_id,
           thrust::device_vector<W> edge_weight)
        : row_ptr_(n),
          col_idx_(edge_index.size()),
          edge_id_(std::move(edge_id)),
          edge_weight_(std::move(edge_weight)),
          bucket_edge_weight_(edge_index.size()),
          opt_(true, true)
    {
        thrust::device_vector<thrust::tuple<T, T, T, W>> to_sort(
            edge_index.size());
        zip(to_sort, edge_index, edge_id_, edge_weight_);
        thrust::sort(to_sort.begin(), to_sort.end());
        thrust::device_vector<T> row_idx_(edge_index.size());
        thrust::device_vector<T> row_ptr_next(edge_index.size());
        unzip(to_sort, row_idx_, col_idx_, edge_id_, edge_weight_);
        handle_partition(row_idx_, local_map_, remote_map_);
        thrust::sequence(row_ptr_.begin(), row_ptr_.end());
        thrust::sequence(row_ptr_next.begin(), row_ptr_next.end());
        thrust::lower_bound(row_idx_.begin(), row_idx_.end(), row_ptr_.begin(),
                            row_ptr_.end(), row_ptr_.begin());
        thrust::upper_bound(row_idx_.begin(), row_idx_.end(),
                            row_ptr_next.begin(), row_ptr_next.end(),
                            row_ptr_next.begin());
        bucket_weight(row_ptr_, row_ptr_next, edge_weight_,
                      bucket_edge_weight_);
    }

    virtual ~quiver() = default;

    size_t size() const { return row_ptr_.size(); }

    size_t edge_counts() const { return col_idx_.size(); }

    sample_option get_option() const { return opt_; }

    const std::unordered_map<T, T> *get_remote_map() const { return &remote_map_; }

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
            sample_functor<T, W>(
                thrust::raw_pointer_cast(row_ptr_.data()), row_ptr_.size(),
                thrust::raw_pointer_cast(col_idx_.data()),
                thrust::raw_pointer_cast(edge_id_.data()),
                thrust::raw_pointer_cast(bucket_edge_weight_.data()),
                col_idx_.size(), thrust::raw_pointer_cast(output_begin),
                thrust::raw_pointer_cast(output_id_begin), opt_.weighted));
    }
};
}  // namespace quiver
