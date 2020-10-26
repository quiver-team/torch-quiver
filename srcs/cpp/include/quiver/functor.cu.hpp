#pragma once
#include <thrust/tuple.h>

template <typename T>
class cap_by
{
    const T cap;

  public:
    cap_by(const T cap) : cap(cap) {}

    __host__ __device__ T operator()(T x) const
    {
        if (x > cap) { return cap; }
        return x;
    }
};

template <typename T>
class value_at
{
    const T *x;

  public:
    value_at(const T *x) : x(x) {}

    __host__ __device__ T operator()(size_t i) const { return x[i]; }
};

template <size_t i>
struct thrust_get {
    template <typename T>
    __device__ auto operator()(const T &t) const
    {
        return thrust::get<i>(t);
    }
};
