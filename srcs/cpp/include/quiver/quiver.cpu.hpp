#pragma once
#include <algorithm>
#include <random>

#include <quiver/quiver.hpp>
#include <quiver/sparse.hpp>
#include <quiver/zip.hpp>

#include <torch/extension.h>

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
class quiver<T, CPU> : public Quiver
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

    virtual ~quiver() = default;

    size_t size() const override { return row_ptr_.size(); }

    size_t edge_counts() const override { return col_idx_.size(); }

    device_t device() const override { return CPU; }

    void sample(const std::vector<int> &vertices, int k) const override
    {
        std::vector<T> inputs(vertices.size());
        std::copy(vertices.begin(), vertices.end(), inputs.begin());
        std::vector<T> outputs(vertices.size() * k);
        std::vector<T> output_counts(vertices.size());

        const T n = row_ptr_.size();
        const T m = col_idx_.size();
        const size_t batch_size = vertices.size();
        for (size_t i = 0; i < batch_size; ++i) {
            T v = vertices[i];
            T begin = row_ptr_[v];
            const T end = v + 1 < n ? row_ptr_[v + 1] : m;
            output_counts[i] =
                safe_sample(col_idx_.data() + begin, col_idx_.data() + end, k,
                            outputs.data() + i * k);
        }
    }

    std::tuple<torch::Tensor, torch::Tensor>
    sample(const torch::Tensor &vertices, int k) const
    {
        const size_t bs = vertices.size(0);
        T *inputs = vertices.data_ptr<T>();
        std::vector<T> outputs(k * bs);
        std::vector<T> output_counts(bs);

        const T n = row_ptr_.size();
        const T m = col_idx_.size();
        size_t total = 0;
        for (size_t i = 0; i < bs; ++i) {
            T v = inputs[i];
            T begin = row_ptr_[v];
            const T end = v + 1 < n ? row_ptr_[v + 1] : m;
            output_counts[i] =
                safe_sample(col_idx_.data() + begin, col_idx_.data() + end, k,
                            outputs.data() + total);
            total += output_counts[i];
        }
        outputs.resize(total);
        torch::Tensor out = torch::empty(total, vertices.options());
        torch::Tensor counts = torch::empty(bs, vertices.options());
        std::copy(outputs.begin(), outputs.end(), out.data_ptr<T>());
        std::copy(output_counts.begin(), output_counts.end(),
                  counts.data_ptr<T>());
        return std::make_tuple(out, counts);
    }
};
}  // namespace quiver
