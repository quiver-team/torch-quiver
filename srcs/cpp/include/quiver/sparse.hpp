#pragma once
#include <stdexcept>
#include <vector>

namespace quiver
{
template <typename T>
bool compress_row_idx(const std::vector<T> &row_idx, std::vector<T> &row_ptr)
{
    const T n = row_ptr.size();
    const T m = row_idx.size();
    T offset = 0;
    for (T i = 0; i < n; ++i) {
        T j = offset;
        for (; j < m; ++j) {
            if (row_idx[j] != i) { break; }
        }
        row_ptr[i] = offset;
        offset = j;
    }
    return offset == m;
}

template <typename T>
std::vector<T> compress_row_idx(const T n, const std::vector<T> &row_idx)
{
    std::vector<T> row_ptr(n);
    if (!compress_row_idx(row_idx, row_ptr)) {
        throw std::invalid_argument("row_idx is not sorted");
    }
    return row_ptr;
}
}  // namespace quiver
