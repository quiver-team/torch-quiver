#pragma once
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>

#include <quiver/functor.cu.hpp>
#include <quiver/trace.hpp>

template <typename T>
class reorder_functor
{
    const T *output_sum;
    const T *counts;
    const T *values;
    T *output;

  public:
    reorder_functor(const T *output_sum, const T *counts, const T *values,
                    T *output)
        : output_sum(output_sum), counts(counts), values(values), output(output)
    {
    }

    __device__ void operator()(const thrust::tuple<T, T> &t)
    {
        const size_t pos = thrust::get<0>(t);
        size_t beg = thrust::get<1>(t);
        size_t output_beg = output_sum[pos];
        const size_t count = counts[pos];
        for (int j = 0; j < count; j++) {
            output[output_beg++] = values[beg++];
        }
    }
};

template <typename T>
void reorder_output(const thrust::device_vector<T> &p1,
                    const thrust::device_vector<T> &p2,
                    const thrust::device_vector<T> &order,
                    const thrust::device_vector<T> &c,
                    const thrust::device_vector<T> &v,
                    thrust::device_vector<T> &out, cudaStream_t stream)
{
    const auto policy = thrust::cuda::par.on(stream);
    auto beg = thrust::make_zip_iterator(
        thrust::make_tuple(order.begin(), p1.begin()));
    auto end =
        thrust::make_zip_iterator(thrust::make_tuple(order.end(), p1.end()));
    thrust::for_each(policy, beg, end,
                     reorder_functor<T>(thrust::raw_pointer_cast(p2.data()),
                                        thrust::raw_pointer_cast(c.data()),
                                        thrust::raw_pointer_cast(v.data()),
                                        thrust::raw_pointer_cast(out.data())));
}

template <typename T>
__global__ void mask_permutation_kernel_1(const size_t n, thrust::pair<T, T> *q,
                                          const size_t m, const T *p)
{
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int worker_count = gridDim.x * blockDim.x;
    for (int i = worker_idx; i < n; i += worker_count) {
        q[i].first = m;
        q[i].second = i;
    }
}

template <typename T>
__global__ void mask_permutation_kernel_2(const size_t n, thrust::pair<T, T> *q,
                                          const size_t m, const T *p)
{
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int worker_count = gridDim.x * blockDim.x;
    for (int i = worker_idx; i < m; i += worker_count) { q[p[i]].first = i; }
}

template <typename T>
void complete_permutation(thrust::device_vector<T> &p, size_t n,
                          cudaStream_t stream)
{
    const size_t m = p.size();
    const auto policy = thrust::cuda::par.on(stream);
    thrust::device_vector<thrust::pair<T, T>> q(n);
    mask_permutation_kernel_1<<<1024, 16, 0, stream>>>(
        n, thrust::raw_pointer_cast(q.data()), m,
        thrust::raw_pointer_cast(p.data()));
    mask_permutation_kernel_2<<<1024, 16, 0, stream>>>(
        n, thrust::raw_pointer_cast(q.data()), m,
        thrust::raw_pointer_cast(p.data()));
    thrust::sort(policy, q.begin(), q.end());
    p.resize(n);
    thrust::transform(policy, q.begin(), q.end(), p.begin(), thrust_get<1>());
}

// given a permutation q of {0, ..., n - 1}, find q, the inverse of p, such that
// q[p[i]] == i for i in {0, ..., n - 1}
template <typename T>
void inverse_permutation(const thrust::device_vector<T> &p,
                         thrust::device_vector<T> &q, cudaStream_t stream)
{
    const size_t n = p.size();
    q.resize(n);
    using it = thrust::counting_iterator<T>;
    thrust::for_each(thrust::cuda::par.on(stream), it(0), it(n),
                     [p = thrust::raw_pointer_cast(p.data()),
                      q = thrust::raw_pointer_cast(q.data())]  //
                     __device__(T i) { q[p[i]] = i; });
}

template <typename T>
void permute_value(const thrust::device_vector<T> &p,
                   thrust::device_vector<T> &a, cudaStream_t stream)
{
    const auto policy = thrust::cuda::par.on(stream);
    thrust::transform(policy, a.begin(), a.end(), a.begin(),
                      value_at<T>(thrust::raw_pointer_cast(p.data())));
}

template <typename T>
thrust::device_vector<T> permute(const thrust::device_vector<T> &p,
                                 const thrust::device_vector<T> &a,
                                 cudaStream_t stream)
{
    const size_t n = a.size();
    thrust::device_vector<T> b(n);
    using it = thrust::counting_iterator<T>;
    thrust::for_each(thrust::cuda::par.on(stream), it(0), it(n),
                     [p = thrust::raw_pointer_cast(p.data()),
                      a = thrust::raw_pointer_cast(a.data()),
                      b = thrust::raw_pointer_cast(b.data())]  //
                     __device__(T i) { b[i] = a[p[i]]; });
    return b;
}

template <typename T>
void _reindex_with(const thrust::device_vector<T> &a,
                   const thrust::device_vector<T> &I,
                   thrust::device_vector<T> &c)
{
    c.resize(a.size());
    thrust::lower_bound(I.begin(), I.end(), a.begin(), a.end(), c.begin());
}

template <typename T, typename P>
void _reindex_with(const P &policy, const thrust::device_vector<T> &a,
                   const thrust::device_vector<T> &I,
                   thrust::device_vector<T> &c)
{
    c.resize(a.size());
    thrust::lower_bound(policy, I.begin(), I.end(), a.begin(), a.end(),
                        c.begin());
}

template <typename T>
void reindex(const thrust::device_vector<T> &a, thrust::device_vector<T> &b,
             thrust::device_vector<T> &c)
{
    {
        b.resize(a.size());
        thrust::copy(a.begin(), a.end(), b.begin());
        thrust::sort(b.begin(), b.end());
        const size_t m = thrust::unique(b.begin(), b.end()) - b.begin();
        b.resize(m);
    }
    _reindex_with(a, b, c);
}

template <typename T>
void reindex_with_seeds(const thrust::device_vector<T> &a,
                        const thrust::device_vector<T> &s,
                        thrust::device_vector<T> &b,
                        thrust::device_vector<T> &c)
{
    TRACE_SCOPE("reindex_with_seeds<thrust>");

    // reindex(a, b, c);
    {
        TRACE_SCOPE("reindex_with_seeds<thrust>::sort unique");

        b.resize(a.size() + s.size());
        thrust::copy(a.begin(), a.end(), b.begin());
        thrust::copy(s.begin(), s.end(), b.begin() + a.size());
        thrust::sort(b.begin(), b.end());
        const size_t m = thrust::unique(b.begin(), b.end()) - b.begin();
        b.resize(m);
    }
    {
        TRACE_SCOPE("reindex_with_seeds<thrust>::_reindex_with 1");
        _reindex_with(a, b, c);
    }

    thrust::device_vector<T> s1;
    s1.reserve(b.size());
    {
        TRACE_SCOPE("reindex_with_seeds<thrust>::_reindex_with 2");
        _reindex_with(s, b, s1);
    }
    {
        TRACE_SCOPE("reindex_with_seeds<thrust>::complete_permutation");
        complete_permutation(s1, b.size());
    }

    {
        TRACE_SCOPE("reindex_with_seeds<thrust>::permute");
        b = permute(s1, b);
    }

    thrust::device_vector<T> s2;
    {
        TRACE_SCOPE("reindex_with_seeds<thrust>::inverse_permutation");
        inverse_permutation(s1, s2);
    }
    {
        TRACE_SCOPE("reindex_with_seeds<thrust>::permute_value");
        permute_value(s2, c);
    }
}

template <typename T>
void reindex(const std::vector<T> &a, std::vector<T> &b, std::vector<T> &c)
{
    thrust::device_vector<T> cuda_a(a.size());
    thrust::copy(a.begin(), a.end(), cuda_a.begin());

    thrust::device_vector<T> cuda_b;
    thrust::device_vector<T> cuda_c;
    reindex(cuda_a, cuda_b, cuda_c);

    b.resize(cuda_b.size());
    thrust::copy(cuda_b.begin(), cuda_b.end(), b.begin());
    c.resize(cuda_c.size());
    thrust::copy(cuda_c.begin(), cuda_c.end(), c.begin());
}

template <typename T>
void reindex_with_seeds(const std::vector<T> &a, const std::vector<T> &s,
                        std::vector<T> &b, std::vector<T> &c)
{
    thrust::device_vector<T> cuda_a(a.size());
    thrust::device_vector<T> cuda_s(s.size());
    {
        TRACE_SCOPE("reindex_with_seeds::copy1");
        thrust::copy(a.begin(), a.end(), cuda_a.begin());
        thrust::copy(s.begin(), s.end(), cuda_s.begin());
    }

    thrust::device_vector<T> cuda_b;
    thrust::device_vector<T> cuda_c;
    reindex_with_seeds(cuda_a, cuda_s, cuda_b, cuda_c);

    {
        TRACE_SCOPE("reindex_with_seeds::copy2");
        b.resize(cuda_b.size());
        thrust::copy(cuda_b.begin(), cuda_b.end(), b.begin());
        c.resize(cuda_c.size());
        thrust::copy(cuda_c.begin(), cuda_c.end(), c.begin());
    }
}

template <typename T>
void reindex_with_seeds(size_t l, const T *s, size_t r, const T *a,
                        std::vector<T> &b, std::vector<T> &c)
{
    thrust::device_vector<T> cuda_a(r);
    thrust::device_vector<T> cuda_s(l);
    {
        TRACE_SCOPE("reindex_with_seeds::copy1");

        thrust::copy(a, a + r, cuda_a.begin());
        thrust::copy(s, s + l, cuda_s.begin());
    }

    thrust::device_vector<T> cuda_b;
    thrust::device_vector<T> cuda_c;
    reindex_with_seeds(cuda_a, cuda_s, cuda_b, cuda_c);

    {
        TRACE_SCOPE("reindex_with_seeds::copy2");
        b.resize(cuda_b.size());
        thrust::copy(cuda_b.begin(), cuda_b.end(), b.begin());
        c.resize(cuda_c.size());
        thrust::copy(cuda_c.begin(), cuda_c.end(), c.begin());
    }
}
