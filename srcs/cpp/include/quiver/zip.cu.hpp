#pragma once

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

namespace quiver {
struct zip_id_functor {
  template <typename T> __device__ void operator()(T tup) {
    thrust::get<0>(tup) =
        thrust::make_tuple(thrust::get<1>(tup).first,
                           thrust::get<1>(tup).second, thrust::get<2>(tup));
  }
};

struct zip_id_weight_functor {
  template <typename T> __device__ void operator()(T tup) {
    thrust::get<0>(tup) = thrust::make_tuple(
        thrust::get<1>(tup).first, thrust::get<1>(tup).second,
        thrust::get<2>(tup), thrust::get<3>(tup));
  }
};

template <typename T>
void unzip(const thrust::device_vector<thrust::pair<T, T>> &p,
           thrust::device_vector<T> &x, thrust::device_vector<T> &y) {
  thrust::transform(p.begin(), p.end(), x.begin(), thrust_get<0>());
  thrust::transform(p.begin(), p.end(), y.begin(), thrust_get<1>());
}

template <typename Ty, typename Tx>
thrust::device_vector<Ty> to_device(const std::vector<Tx> &x) {
  static_assert(sizeof(Tx) == sizeof(Ty), "");
  thrust::device_vector<Ty> y(x.size());
  thrust::copy(reinterpret_cast<const Ty *>(x.data()),
               reinterpret_cast<const Ty *>(x.data()) + x.size(), y.begin());
  return std::move(y);
}

template <typename T>
void zip(thrust::device_vector<thrust::tuple<T, T, T>> &t,
         const thrust::device_vector<thrust::pair<T, T>> &x,
         const thrust::device_vector<T> &y) {
  thrust::for_each(
      thrust::make_zip_iterator(
          thrust::make_tuple(t.begin(), x.begin(), y.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(t.end(), x.end(), y.end())),
      zip_id_functor());
}

template <typename T>
void unzip(const thrust::device_vector<thrust::tuple<T, T, T>> &t,
           thrust::device_vector<T> &x, thrust::device_vector<T> &y,
           thrust::device_vector<T> &z) {
  thrust::transform(t.begin(), t.end(), x.begin(), thrust_get<0>());
  thrust::transform(t.begin(), t.end(), y.begin(), thrust_get<1>());
  thrust::transform(t.begin(), t.end(), z.begin(), thrust_get<2>());
}

template <typename T, typename W>
void zip(thrust::device_vector<thrust::tuple<T, T, T, W>> &t,
         const thrust::device_vector<thrust::pair<T, T>> &x,
         const thrust::device_vector<T> &y, const thrust::device_vector<W> &z) {
  thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                       t.begin(), x.begin(), y.begin(), z.begin())),
                   thrust::make_zip_iterator(
                       thrust::make_tuple(t.end(), x.end(), y.end(), z.end())),
                   zip_id_weight_functor());
}

template <typename T, typename W>
void unzip(const thrust::device_vector<thrust::tuple<T, T, T, W>> &t,
           thrust::device_vector<T> &x, thrust::device_vector<T> &y,
           thrust::device_vector<T> &z, thrust::device_vector<W> &w) {
  thrust::transform(t.begin(), t.end(), x.begin(), thrust_get<0>());
  thrust::transform(t.begin(), t.end(), y.begin(), thrust_get<1>());
  thrust::transform(t.begin(), t.end(), z.begin(), thrust_get<2>());
  thrust::transform(t.begin(), t.end(), w.begin(), thrust_get<3>());
}
} // namespace quiver