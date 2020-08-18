#pragma once
#include <thrust/pair.h>

template <int i>
struct thrust_get {
    template <typename T>
    __device__ auto operator()(const T &t) const
    {
        return thrust::get<i>(t);
    }
};
