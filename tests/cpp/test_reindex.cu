#include <algorithm>
#include <random>
#include <vector>

#include "testing.hpp"
#include <quiver/reindex.cu.hpp>

template <typename T>
std::pair<std::vector<T>, std::vector<T>> gen_inverse_permutation_pair(size_t n)
{
    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<T> x(n);
    std::vector<T> y(n);
    std::iota(x.begin(), x.end(), 0);
    std::shuffle(x.begin(), x.end(), g);
    for (size_t i = 0; i < n; ++i) { y[x[i]] = i; }
    return std::make_pair(std::move(x), std::move(y));
}

template <typename T>
void test_inverse_permutation(size_t n)
{
    auto xy = gen_inverse_permutation_pair<T>(n);
    std::vector<T> x, y;
    std::tie(x, y) = std::move(xy);

    std::vector<T> z(n);

    thrust::device_vector<T> p(n);
    thrust::device_vector<T> q(n);

    thrust::copy(x.begin(), x.end(), p.begin());
    inverse_permutation(p, q, 0);
    thrust::copy(q.begin(), q.end(), z.begin());

    ASSERT_EQ(y, z);
}

TEST(test_reindex, test_inverse_permutation_1)
{
    test_inverse_permutation<int>(5);
    test_inverse_permutation<int>(10);
    test_inverse_permutation<int>(15);
    test_inverse_permutation<int>(20);
    test_inverse_permutation<int>(100);
    test_inverse_permutation<int>(1000);
    test_inverse_permutation<int>(10000);
}

template <typename T>
void test_complete_permutation(size_t n)
{
    size_t m = n / 2;
    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<T> x(n);
    std::iota(x.begin(), x.end(), 0);
    std::shuffle(x.begin(), x.end(), g);
    std::sort(x.begin() + m, x.end());

    thrust::device_vector<T> p(m);
    thrust::copy(x.begin(), x.begin() + m, p.begin());
    complete_permutation(p, n, 0);
    std::vector<T> y(n);
    thrust::copy(p.begin(), p.end(), y.begin());

    ASSERT_EQ(x, y);
}

TEST(test_reindex, test_complete_permutation_1)
{
    test_complete_permutation<int>(5);
    test_complete_permutation<int>(10);
    test_complete_permutation<int>(15);
    test_complete_permutation<int>(20);
    test_complete_permutation<int>(100);
    test_complete_permutation<int>(1000);
    test_complete_permutation<int>(10000);
}
