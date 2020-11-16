#pragma once
#include <quiver/algorithm.cu.hpp>
#include <quiver/common.hpp>
#include <quiver/copy.cu.hpp>
#include <quiver/cuda_random.cu.hpp>
#include <quiver/functor.cu.hpp>
#include <quiver/quiver.hpp>
#include <quiver/trace.hpp>
#include <quiver/zip.cu.hpp>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>

#include <cuda_runtime.h>

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

    __host__ __device__ T operator()(T i) const
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
void handle_partition(thrust::device_vector<T> &row_idx,
                      thrust::device_vector<T> &local_map_)
{
    local_map_.resize(row_idx.size());
    auto last =
        thrust::unique_copy(row_idx.begin(), row_idx.end(), local_map_.begin());
    int len = last - local_map_.begin();
    local_map_.resize(len);
    thrust::lower_bound(local_map_.begin(), local_map_.end(), row_idx.begin(),
                        row_idx.end(), row_idx.begin());
}

template <typename T>
class quiver<T, CUDA>
{
    using self = quiver<T, CUDA>;
    using W = float;

    const thrust::device_vector<T> row_ptr_;
    const thrust::device_vector<T> col_idx_;
    const thrust::device_vector<T> edge_idx_;  // TODO: make it optional

    const thrust::device_vector<W> edge_weight_;         // optional
    const thrust::device_vector<W> bucket_edge_weight_;  // optional

    const thrust::device_vector<T> local_map_;

    const sample_option opt_;

    quiver(thrust::device_vector<T> local_map, thrust::device_vector<T> row_ptr,
           thrust::device_vector<T> col_idx, thrust::device_vector<T> edge_idx)
        : local_map_(std::move(local_map)),
          row_ptr_(std::move(row_ptr)),
          col_idx_(std::move(col_idx)),
          edge_idx_(std::move(edge_idx)),
          opt_(false, true)
    {
    }

    quiver(thrust::device_vector<T> local_map, thrust::device_vector<T> row_ptr,
           thrust::device_vector<T> col_idx, thrust::device_vector<T> edge_idx,
           thrust::device_vector<W> edge_weight,
           thrust::device_vector<W> bucket_edge_weight)
        : local_map_(std::move(local_map)),
          row_ptr_(std::move(row_ptr)),
          col_idx_(std::move(col_idx)),
          edge_idx_(std::move(edge_idx)),
          edge_weight_(std::move(edge_weight)),
          bucket_edge_weight_(std::move(bucket_edge_weight)),
          opt_(true, true)
    {
    }

  public:
    static self New(T n, thrust::device_vector<T> row_idx,
                    thrust::device_vector<T> col_idx,
                    thrust::device_vector<T> edge_idx)
    {
        thrust::device_vector<thrust::tuple<T, T, T>> edges(edge_idx.size());
        zip(row_idx, col_idx, edge_idx, edges);
        thrust::sort(edges.begin(), edges.end());
        unzip(edges, row_idx, col_idx, edge_idx);

        thrust::device_vector<T> local_map;
        handle_partition(row_idx, local_map);

        thrust::device_vector<T> row_ptr(n);
        thrust::sequence(row_ptr.begin(), row_ptr.end());
        thrust::lower_bound(row_idx.begin(), row_idx.end(), row_ptr.begin(),
                            row_ptr.end(), row_ptr.begin());
        return self(local_map, row_ptr, col_idx, edge_idx);
    }

    static self New(T n, thrust::device_vector<T> row_idx,
                    thrust::device_vector<T> col_idx,
                    thrust::device_vector<T> edge_idx,
                    thrust::device_vector<W> edge_weight)
    {
        thrust::device_vector<thrust::tuple<T, T, T, W>> edges(edge_idx.size());
        zip(row_idx, col_idx, edge_idx, edge_weight, edges);
        thrust::sort(edges.begin(), edges.end());
        unzip(edges, row_idx, col_idx, edge_idx, edge_weight);

        thrust::device_vector<T> local_map;
        handle_partition(row_idx, local_map);

        thrust::device_vector<T> row_ptr(n);
        thrust::device_vector<T> row_ptr_next(edge_idx.size());
        thrust::device_vector<W> bucket_edge_weight(edge_idx.size());
        thrust::sequence(row_ptr.begin(), row_ptr.end());
        thrust::sequence(row_ptr_next.begin(), row_ptr_next.end());
        thrust::lower_bound(row_idx.begin(), row_idx.end(), row_ptr.begin(),
                            row_ptr.end(), row_ptr.begin());
        thrust::upper_bound(row_idx.begin(), row_idx.end(),
                            row_ptr_next.begin(), row_ptr_next.end(),
                            row_ptr_next.begin());
        bucket_weight(row_ptr, row_ptr_next, edge_weight, bucket_edge_weight);
        return self(local_map, row_ptr, col_idx, edge_idx, edge_weight,
                    bucket_edge_weight);
    }

    virtual ~quiver() = default;

    size_t size() const { return row_ptr_.size(); }

    size_t edge_counts() const { return col_idx_.size(); }

    sample_option get_option() const { return opt_; }

    void to_local(const cudaStream_t stream,
                  thrust::device_vector<T> &ids) const
    {
        // TODO: skip if local_map is the full set
        thrust::lower_bound(thrust::cuda::par.on(stream), local_map_.begin(),
                            local_map_.end(), ids.begin(), ids.end(),
                            ids.begin());
    }

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

    void async_degree(const cudaStream_t stream,
                      thrust::device_ptr<const T> input_begin,
                      thrust::device_ptr<const T> input_end,
                      thrust::device_ptr<T> output_begin) const
    {
        async_transform(
            kernal_option(stream), input_begin, input_end, output_begin,
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
                thrust::raw_pointer_cast(edge_idx_.data()),
                thrust::raw_pointer_cast(bucket_edge_weight_.data()),
                col_idx_.size(), thrust::raw_pointer_cast(output_begin),
                thrust::raw_pointer_cast(output_id_begin), opt_.weighted));
    }

    template <typename Iter>
    void async_sample(const cudaStream_t stream, Iter input_begin,
                      Iter input_end, Iter output_ptr_begin,
                      Iter output_count_begin,
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
        async_for_each(
            kernal_option(stream), begin, end,
            sample_functor<T, W>(
                thrust::raw_pointer_cast(row_ptr_.data()), row_ptr_.size(),
                thrust::raw_pointer_cast(col_idx_.data()),
                thrust::raw_pointer_cast(edge_idx_.data()),
                thrust::raw_pointer_cast(bucket_edge_weight_.data()),
                col_idx_.size(), thrust::raw_pointer_cast(output_begin),
                thrust::raw_pointer_cast(output_id_begin), opt_.weighted));
    }

#ifdef QUIVER_TEST
    void get_edges(std::vector<T> &u, std::vector<T> &v) const
    {
        const auto row_idx = from_device<T>(local_map_);
        const auto row_ptr = from_device<T>(row_ptr_);
        const auto col_idx = from_device<T>(col_idx_);

        const int n = row_ptr.size();
        const int m = col_idx.size();
        u.clear();
        v.clear();
        u.reserve(m);
        v.reserve(m);

        get_adj_diff<T> count(thrust::raw_pointer_cast(row_ptr.data()), n, m);

        for (int i = 0; i < n; ++i) {
            const int c = count(i);
            for (int j = 0; j < c; ++j) {
                u.push_back(row_idx.at(i));
                v.push_back(col_idx[row_ptr.at(i) + j]);
            }
        }
    }
#endif
};
}  // namespace quiver
