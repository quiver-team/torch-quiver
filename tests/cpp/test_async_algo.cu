#include <quiver/algorithm.cu.hpp>
#include <quiver/cuda_stream.hpp>
#include <thrust/device_vector.h>

#include "testing.hpp"

__device__ int fibo(int x)
{
    if (x < 2) { return 1; }
    return fibo(x - 1) + fibo(x - 2);
}

void test_transform_1(int n)
{
    thrust::device_vector<int> x(n);
    thrust::device_vector<int> y(n);
    thrust::device_vector<int> z(n);

    thrust::sequence(x.begin(), x.end());
    thrust::fill(y.begin(), y.end(), -1234);
    thrust::fill(z.begin(), z.end(), -4321);

    const auto f = [] __device__(int i) { return fibo(i % 17) % 5; };

    thrust::transform(x.begin(), x.end(), y.begin(), f);

    quiver::cuda_stream stream;
    quiver::async_transform(quiver::kernal_option(stream), x.begin(), x.end(),
                            z.begin(), f);

    bool ok = thrust::equal(y.begin(), y.end(), z.begin());
    ASSERT_TRUE(ok);
}

TEST(test_async, test_async_transform_1)
{
    const auto ns = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1 << 10, 1 << 20,
    };
    for (auto n : ns) { test_transform_1(n); }
}

struct F {
    int *p;

    F(thrust::device_ptr<int> p) : p(thrust::raw_pointer_cast(p)) {}

    __device__ void operator()(int i) { p[i] = fibo(i % 17) % 5; }
};

void test_for_each_1(int n)
{
    thrust::device_vector<int> x(n);
    thrust::device_vector<int> y(n);
    thrust::device_vector<int> z(n);

    thrust::sequence(x.begin(), x.end());
    thrust::fill(y.begin(), y.end(), -1234);
    thrust::fill(z.begin(), z.end(), -4321);

    thrust::for_each(x.begin(), x.end(), F(y.data()));

    quiver::cuda_stream stream;
    quiver::async_for_each(quiver::kernal_option(stream), x.begin(), x.end(),
                           F(z.data()));

    bool ok = thrust::equal(y.begin(), y.end(), z.begin());
    ASSERT_TRUE(ok);
}

TEST(test_async, test_async_for_each_1)
{
    const auto ns = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1 << 10, 1 << 20,
    };
    for (auto n : ns) { test_for_each_1(n); }
}
