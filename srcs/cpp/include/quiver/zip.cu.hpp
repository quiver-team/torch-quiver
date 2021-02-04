#pragma once
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

namespace quiver
{
template <typename... T>
struct zipper {
    using Tuple = thrust::tuple<T...>;

    template <typename... X>
    void operator()(thrust::device_vector<Tuple> &t, const X &... xs)
    {
        thrust::copy(
            thrust::make_zip_iterator(thrust::make_tuple(xs.begin()...)),
            thrust::make_zip_iterator(thrust::make_tuple(xs.end()...)),
            t.begin());
    }
};

template <typename T0, typename T1>
void zip(const thrust::device_vector<T0> &x, const thrust::device_vector<T1> &y,
         thrust::device_vector<thrust::tuple<T0, T1>> &t)
{
    zipper<T0, T1>()(t, x, y);
}

template <typename T0, typename T1>
void unzip(const thrust::device_vector<thrust::tuple<T0, T1>> &t,
           thrust::device_vector<T0> &x, thrust::device_vector<T1> &y)
{
    thrust::transform(t.begin(), t.end(), x.begin(), thrust_get<0>());
    thrust::transform(t.begin(), t.end(), y.begin(), thrust_get<1>());
}

template <typename T0, typename T1, typename T2>
void zip(const thrust::device_vector<T0> &x, const thrust::device_vector<T1> &y,
         const thrust::device_vector<T2> &z,
         thrust::device_vector<thrust::tuple<T0, T1, T2>> &t)
{
    zipper<T0, T1, T2>()(t, x, y, z);
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
    zipper<T0, T1, T2, T3>()(t, x, y, z, w);
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
