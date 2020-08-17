#include <quiver/quiver.cpu.hpp>
#include <quiver/quiver.hpp>

namespace quiver
{
Quiver::Quiver() {}

std::unique_ptr<Quiver>
Quiver::from_edge_index_cpu(int n, std::vector<std::pair<int, int>> edge_index)
{
    using T = int64_t;
    using Q = quiver<T, CPU>;
    using vec = std::vector<std::pair<T, T>>;
    vec ei(edge_index.size());
    std::copy(edge_index.begin(), edge_index.end(), ei.begin());
    edge_index.clear();
    return std::make_unique<Q>(n, std::move(ei));
}

std::unique_ptr<Quiver>
Quiver::from_edge_index(int n, std::vector<std::pair<int, int>> edge_index,
                        device_t device)
{
    if (device == CUDA) {
#if HAVE_CUDA
        return from_edge_index_cuda(n, std::move(edge_index));
#else
        fprintf(stderr, "Not built with CUDA support\n");
        return std::unique_ptr<Quiver>();
#endif
    }
    return from_edge_index_cpu(n, std::move(edge_index));
}
}  // namespace quiver
