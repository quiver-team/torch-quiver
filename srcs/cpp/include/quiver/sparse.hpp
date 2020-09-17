#pragma once
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace quiver
{
template <typename T>
std::vector<T> compress_row_idx(const T n, const std::vector<T> &sorted_row_idx)
{
    const T m = sorted_row_idx.size();
    std::vector<T> r(n);
    T offset = 0;
    for (T i = 0; i < n; ++i) {
        T j = offset;
        for (; j < m; ++j) {
            if (sorted_row_idx[j] != i) { break; }
        }
        r[i] = offset;
        offset = j;
    }
    if (offset != m) { throw std::invalid_argument("row_idx is not sorted"); }
    return r;
}
}  // namespace quiver
