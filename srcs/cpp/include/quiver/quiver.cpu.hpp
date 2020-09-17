#pragma once
#include <algorithm>
#include <random>

#include <quiver/quiver.hpp>
#include <quiver/zip.hpp>

namespace quiver
{
template <typename T>
void cpu_sample(const T *begin, const T *end, const int k, T *outputs,
                T *output_count)
{
    const T cap = end - begin;
    if (cap <= k) {
        *output_count = cap;
        std::copy(begin, end, outputs);
    } else {
        *output_count = k;
        thread_local static std::random_device device;
        thread_local static std::mt19937 g(device());
        std::sample(begin, end, outputs, k, g);
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
            cpu_sample(col_idx_.data() + begin, col_idx_.data() + end, k,
                       outputs.data() + i * k, output_counts.data() + i);
        }
    }
};
}  // namespace quiver
