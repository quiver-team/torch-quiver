#pragma once
#include <algorithm>
#include <memory>
#include <vector>

namespace quiver
{
enum device_t {
    CPU,
    CUDA,
};

inline const char *device_name(device_t device)
{
    switch (device) {
    case CPU:
        return "CPU";
    case CUDA:
        return "CUDA";
    default:
        throw std::invalid_argument("invalid device");
    }
}

class Quiver
{
    static std::unique_ptr<Quiver>
    from_edge_index_cpu(int n, std::vector<std::pair<int, int>> edge_index);

    static std::unique_ptr<Quiver>
    from_edge_index_cuda(int n, std::vector<std::pair<int, int>> edge_index);

  public:
    Quiver();
    virtual ~Quiver() = default;

    virtual void sample(const std::vector<int> &vertices, int k) const = 0;

    virtual size_t size() const = 0;
    virtual size_t edge_counts() const = 0;
    virtual device_t device() const = 0;

    static std::unique_ptr<Quiver>
    from_edge_index(int n, std::vector<std::pair<int, int>> edge_index,
                    device_t device);
};

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

template <typename T>
void zip(const T *begin1, const T *end1, const T *begin2, T *output)
{
    for (; begin1 < end1;) {
        *(output++) = *(begin1++);
        *(output++) = *(begin2++);
    }
}

// template <typename T1, typename T2 = T1>
// std::vector<std::pair<T1, T2>> zip(const std::vector<T1> &a,
//                                    const std::vector<T2> &b)
// {
//     std::vector<T1, T2> c(a.size());
//     return c;
// }

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

template <typename T, device_t device>
class quiver;
}  // namespace quiver
