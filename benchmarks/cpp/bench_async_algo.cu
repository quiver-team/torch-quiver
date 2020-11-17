#include <sstream>

#include <quiver/algorithm.cu.hpp>
#include <quiver/cuda_stream.hpp>
#include <thrust/device_vector.h>

#include <quiver/trace.hpp>

DEFINE_TRACE_CONTEXTS;

__device__ int fibo(int x)
{
    if (x < 2) { return 1; }
    return fibo(x - 1) + fibo(x - 2);
}

void bench_1(int n, int times = 10)
{
    std::stringstream name;
    name << __func__ << "(n=" << n << ", times=" << times << ")";
    TRACE_SCOPE(name.str());

    thrust::device_vector<int> x(n);
    thrust::sequence(x.begin(), x.end());
    thrust::device_vector<int> y(n);

    const auto f = [] __device__(int i) { return fibo(i % 17) % 5; };

    {
        TRACE_SCOPE("sync-" + std::to_string(times) + "-times(" +
                    std::to_string(n) + ")");

        for (int i = 0; i < times; ++i) {
            TRACE_SCOPE("thrust::transform(" + std::to_string(n) + ")");
            thrust::transform(x.begin(), x.end(), y.begin(), f);
        }
    }
    {
        TRACE_SCOPE("async-" + std::to_string(times) + "-times(" +
                    std::to_string(n) + ")");

        std::unique_ptr<quiver::cuda_stream> stream(new quiver::cuda_stream);
        quiver::kernal_option o(*stream);

        thrust::device_vector<int> z(times);
        for (int i = 0; i < times; ++i) {
            TRACE_SCOPE("async_transform(" + std::to_string(n) + ")");
            quiver::async_transform(o, x.begin(), x.end(), y.begin(), f);
        }
        {
            TRACE_SCOPE(name.str() + "::cudaStreamSynchronize");
            cudaStreamSynchronize(*stream);
        }
        {
            TRACE_SCOPE(name.str() + "::stream.reset");
            stream.reset();
        }
    }
}

int main()
{
    TRACE_SCOPE(__func__);

    bench_1(1 << 10);
    bench_1(1 << 20);

    return 0;
}
