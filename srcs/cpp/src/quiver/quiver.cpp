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
}  // namespace quiver
