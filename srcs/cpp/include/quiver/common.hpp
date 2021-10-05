#pragma once
#include <stdexcept>

namespace quiver
{
inline void check(bool f)
{
    if (!f) { throw std::runtime_error(std::string("check failed")); }
}

template <typename T>
void check_eq(const T &x, const T &y)
{
    if (x == y) { return; }
    throw std::runtime_error(std::string("check_eq failed"));
}
enum QuiverMode { DMA, ZERO_COPY, PAGE_MIGRATION };
#define cudaCheckError()                                                       \
    {                                                                          \
        cudaError_t e = cudaGetLastError();                                    \
        if (e != cudaSuccess) {                                                \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,           \
                   cudaGetErrorString(e));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

}  // namespace quiver
