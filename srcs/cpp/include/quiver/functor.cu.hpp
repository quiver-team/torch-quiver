#pragma once

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
