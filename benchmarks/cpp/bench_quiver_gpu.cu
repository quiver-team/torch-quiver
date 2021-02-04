#include <algorithm>
#include <cstdio>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <vector>

#include <thrust/binary_search.h>

#include <quiver/quiver.cu.hpp>
#include <quiver/trace.hpp>

#include "common.hpp"

DEFINE_TRACE_CONTEXTS;

using W = float;
using V = int64_t;
using Quiver = quiver::quiver<V, quiver::CUDA>;

void bench_sample_once(const cudaStream_t stream, const Quiver &q,
                       const std::vector<V> &batch, int k)
{
    TRACE_SCOPE(__func__);
    const auto policy = thrust::cuda::par.on(stream);

    thrust::device_vector<V> inputs(batch.size());
    thrust::device_vector<V> output_counts(batch.size());
    thrust::device_vector<V> output_ptr(batch.size());

    thrust::copy(batch.begin(), batch.end(), inputs.begin());

    q.degree(stream, inputs.data(), inputs.data() + inputs.size(),
             output_counts.data());
    if (k >= 0) {
        thrust::transform(policy, output_counts.begin(), output_counts.end(),
                          output_counts.begin(), cap_by<V>(k));
    }

    thrust::exclusive_scan(policy, output_counts.begin(), output_counts.end(),
                           output_ptr.begin());
    const V tot =
        thrust::reduce(policy, output_counts.begin(), output_counts.end());

    thrust::device_vector<V> outputs(tot);
    thrust::device_vector<V> output_eid(tot);

    {
        TRACE_SCOPE("sample kernel");
        q.sample(stream, inputs.begin(), inputs.end(), output_ptr.begin(),
                 output_counts.begin(), outputs.data(), output_eid.data());
        cudaDeviceSynchronize();
    }
}

void bench_1(const graph &g)
{
    TRACE_SCOPE(__func__);

    std::vector<V> u(g.M());
    std::vector<V> v(g.M());
    std::vector<W> w(g.M());

    thrust::device_vector<V> row_idx = to_device<V>(u);
    thrust::device_vector<V> col_idx = to_device<V>(v);
    thrust::device_vector<W> edge_weight = to_device<W>(w);
    thrust::device_vector<V> edge_idx(g.M());
    thrust::sequence(edge_idx.begin(), edge_idx.end());
    Quiver q = Quiver::New(g.N(), row_idx, col_idx, edge_idx, edge_weight);

    printf("|V|=%d, |E|=%d\n", (int)q.size(), (int)q.edge_counts());

    const int batch_size = 1024;
    std::vector<V> seq(g.N());
    std::iota(seq.begin(), seq.end(), 0);
    std::vector<V> batch(batch_size);

    const int steps = 100;
    const int k = 5;
    for (int i = 0; i < steps; ++i) {
        std_sample_i64(seq.data(), seq.size(), batch.data(), batch.size());
        bench_sample_once(0, q, batch, k);
    }
}

int main(int argc, char *argv[])
{
    TRACE_SCOPE(__func__);
    using W = float;
    const int N = 1000000;
    const int M = 4000000;
    auto g = TRACE_EXPR(gen_random_graph(N, M));
    bench_1(g);

    return 0;
}
