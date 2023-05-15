#pragma once
#include <algorithm>
#include <random>
#include <quiver/common.hpp>
#include <quiver/quiver.hpp>
#include <quiver/sparse.hpp>
#include <quiver/zip.hpp>
#include <ATen/Parallel.h>
#include <iostream>

namespace quiver
{
// sample at most k elements from [begin, end), returns the sampled count.
template <typename T, typename N>
N safe_sample(const T *begin, const T *end, const N k, T *outputs)
{
    const N cap = end - begin;
    if (k < cap) {
        thread_local static std::random_device device;
        thread_local static std::mt19937 g(device());
        std::sample(begin, end, outputs, k, g);
        return k;
    } else {
        std::copy(begin, end, outputs);
        return cap;
    }
}

template <typename T>
class quiver<T, CPU>
{
    std::vector<T> row_ptr_;
    std::vector<T> col_idx_;


  public:
    quiver(T n, std::vector<std::pair<T, T>> edge_index)
        : row_ptr_(n), col_idx_(edge_index.size())
    {
        std::sort(edge_index.begin(), edge_index.end());
        const auto [row_idx, col_idx] = unzip(edge_index);
        std::vector<T> row_ptr = compress_row_idx(n, row_idx);
        std::copy(row_ptr.begin(), row_ptr.end(), row_ptr_.begin());
        std::copy(col_idx.begin(), col_idx.end(), col_idx_.begin());
    }
    quiver(T node_count, T edge_count, T* row_ptr, T* indices){
        row_ptr_ = std::vector<T>(row_ptr, row_ptr + node_count);
        col_idx_ = std::vector<T>(indices, indices + edge_count);
    }

    virtual ~quiver() = default;

    size_t size() const { return row_ptr_.size();}

    size_t edge_counts() const { return col_idx_.size(); }

    std::tuple<std::vector<T>, std::vector<T>>
    sample_kernel(const std::vector<T> &inputs, int k) const
    {
        const size_t bs = inputs.size();
        std::vector<T> outputs;
        std::vector<T> output_counts(bs);
        std::vector<T> output_count_prefix_sum(bs + 1);
        output_count_prefix_sum[0] = 0;

        const T n = row_ptr_.size();
        const T m = col_idx_.size();

        // std::cout<<"get num threads "<< at::get_num_threads()<<std::endl;
        
        at::parallel_for(0, bs, 1, [&](size_t start, size_t end){
            for(size_t i = start; i < end; i++){
                T v = inputs[i];
                T begin = row_ptr_[v];
                const T end = v + 1 < n ? row_ptr_[v + 1] : m;
                output_counts[i] = (end - begin) < k ? end - begin : k;
            }
        });

        {
            // output_count_prefix_sum should be [0, output_counts[0], output_counts[0] + output_counts[1], ]
            for(size_t i = 1; i <= bs; i ++){
                output_count_prefix_sum[i] = output_count_prefix_sum[i - 1] + output_counts[i - 1];
            }

        }

        //std::cout<<"all sampled nodes "<< output_count_prefix_sum[bs]<<std::endl;
        outputs.resize(output_count_prefix_sum[bs]);

        at::parallel_for(0, bs, 1, [&](size_t start, size_t end){
            for(size_t i = start; i < end; ++i) {
                T v = inputs[i];
                T begin = row_ptr_[v];
                const T end = v + 1 < n ? row_ptr_[v + 1] : m;
                
                safe_sample(col_idx_.data() + begin, col_idx_.data() + end, k, outputs.data() + output_count_prefix_sum[i]);
            }

        });
        return std::make_tuple(std::move(outputs), std::move(output_counts));
    }
};
}  // namespace quiver
