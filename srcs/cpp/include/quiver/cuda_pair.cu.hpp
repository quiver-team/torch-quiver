#pragma once

template <typename T1, typename T2 = T1>
struct cuda_pair {
    T1 first;
    T2 second;

    __device__ bool operator<(const cuda_pair &p) const
    {
        return first < p.first || first == p.first && second < p.second;
    }
};
