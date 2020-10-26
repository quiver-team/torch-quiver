#pragma once
#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

template <typename T>
std::vector<T> sorted(const std::vector<T> &x)
{
    std::vector<T> y(x.size());
    std::copy(x.begin(), x.end(), y.begin());
    std::sort(y.begin(), y.end());
    return y;
}

template <typename T>
bool same_content(const std::vector<T> &x, const std::vector<T> &y)
{
    const auto a = sorted(x);
    const auto b = sorted(y);
    return a == b;
}
