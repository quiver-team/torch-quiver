#pragma once
#include <vector>

#include <thrust/device_vector.h>

template <typename Ty, typename Tx>
thrust::device_vector<Ty> to_device(const std::vector<Tx> &x)
{
    static_assert(sizeof(Tx) == sizeof(Ty), "");
    thrust::device_vector<Ty> y(x.size());
    thrust::copy(reinterpret_cast<const Ty *>(x.data()),
                 reinterpret_cast<const Ty *>(x.data()) + x.size(), y.begin());
    return std::move(y);
}

template <typename Ty, typename Tx>
std::vector<Ty> from_device(const thrust::device_vector<Tx> &x)
{
    static_assert(sizeof(Tx) == sizeof(Ty), "");
    std::vector<Ty> y(x.size());
    thrust::copy(x.begin(), x.end(), reinterpret_cast<Tx *>(y.data()));
    return std::move(y);
}
