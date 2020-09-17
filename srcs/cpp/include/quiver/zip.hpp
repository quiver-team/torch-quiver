#pragma once
#include <algorithm>
#include <vector>

namespace quiver
{
template <typename T>
void zip(const T *begin1, const T *end1, const T *begin2, T *output)
{
    for (; begin1 < end1;) {
        *(output++) = *(begin1++);
        *(output++) = *(begin2++);
    }
}

template <typename T>
std::pair<std::vector<T>, std::vector<T>>
unzip(const std::vector<std::pair<T, T>> &z)
{
    std::vector<T> x(z.size());
    std::vector<T> y(z.size());
    std::transform(z.begin(), z.end(), x.begin(),
                   [](auto &p) { return p.first; });
    std::transform(z.begin(), z.end(), y.begin(),
                   [](auto &p) { return p.second; });
    return std::make_pair(std::move(x), std::move(y));
}
}  // namespace quiver
