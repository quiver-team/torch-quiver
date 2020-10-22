#pragma once

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

namespace quiver
{
template <typename T0, typename T1>
void unzip(const thrust::device_vector<thrust::pair<T0, T1>> &p,
           thrust::device_vector<T0> &x, thrust::device_vector<T1> &y)
{
    thrust::transform(p.begin(), p.end(), x.begin(), thrust_get<0>());
    thrust::transform(p.begin(), p.end(), y.begin(), thrust_get<1>());
}

template <typename Ty, typename Tx>
thrust::device_vector<Ty> to_device(const std::vector<Tx> &x)
{
    static_assert(sizeof(Tx) == sizeof(Ty), "");
    thrust::device_vector<Ty> y(x.size());
    thrust::copy(reinterpret_cast<const Ty *>(x.data()),
                 reinterpret_cast<const Ty *>(x.data()) + x.size(), y.begin());
    return std::move(y);
}

template <typename T0, typename T1, typename T2>
void zip(const thrust::device_vector<T0> &x, const thrust::device_vector<T1> &y,
         const thrust::device_vector<T2> &z,
         thrust::device_vector<thrust::tuple<T0, T1, T2>> &t)
{
    thrust::copy(thrust::make_zip_iterator(
                     thrust::make_tuple(x.begin(), y.begin(), z.begin())),
                 thrust::make_zip_iterator(
                     thrust::make_tuple(x.end(), y.end(), z.end())),
                 t.begin());
}

template <typename T0, typename T1, typename T2>
void unzip(const thrust::device_vector<thrust::tuple<T0, T1, T2>> &t,
           thrust::device_vector<T0> &x, thrust::device_vector<T1> &y,
           thrust::device_vector<T2> &z)
{
    thrust::transform(t.begin(), t.end(), x.begin(), thrust_get<0>());
    thrust::transform(t.begin(), t.end(), y.begin(), thrust_get<1>());
    thrust::transform(t.begin(), t.end(), z.begin(), thrust_get<2>());
}

template <typename T0, typename T1, typename T2, typename T3>
void zip(const thrust::device_vector<T0> &x, const thrust::device_vector<T1> &y,
         const thrust::device_vector<T2> &z, const thrust::device_vector<T3> &w,
         thrust::device_vector<thrust::tuple<T0, T1, T2, T3>> &t)
{
    thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(
                     x.begin(), y.begin(), z.begin(), w.begin())),
                 thrust::make_zip_iterator(
                     thrust::make_tuple(x.end(), y.end(), z.end(), w.end())),
                 t.begin());
}

template <typename T0, typename T1, typename T2, typename T3>
void unzip(const thrust::device_vector<thrust::tuple<T0, T1, T2, T3>> &t,
           thrust::device_vector<T0> &x, thrust::device_vector<T1> &y,
           thrust::device_vector<T2> &z, thrust::device_vector<T3> &w)
{
    thrust::transform(t.begin(), t.end(), x.begin(), thrust_get<0>());
    thrust::transform(t.begin(), t.end(), y.begin(), thrust_get<1>());
    thrust::transform(t.begin(), t.end(), z.begin(), thrust_get<2>());
    thrust::transform(t.begin(), t.end(), w.begin(), thrust_get<3>());
}
}  // namespace quiver
