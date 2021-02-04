#pragma once
#include <vector>

#include <quiver/cuda_stream.hpp>

namespace quiver
{
class stream_pool
{
    std::vector<cuda_stream> streams_;

  public:
    stream_pool() {}
    stream_pool(int size) : streams_(size) {}
    cudaStream_t operator[](int i) const
    {
        if (i < 0 || i >= streams_.size()) { return 0; }
        return streams_[i];
    }
    bool empty() const { return streams_.empty(); }
};
}  // namespace quiver
