#include <algorithm>
#include <unordered_set>
#include <vector>

#include <quiver/quiver.cpu.hpp>

#include "testing.hpp"

class simple_graph
{
    int nodes_;
    int neighbor_;
    std::vector<std::pair<int64_t, int64_t>> edges_;

  public:
    simple_graph(int N, int neighbor)
        : nodes_(N), neighbor_(neighbor), edges_(N * neighbor)
    {
        for (int i = 0; i < N; i++) {
            size_t index = i * neighbor;
            for (int j = 0; j < neighbor; j++) {
                int64_t src = i;
                int64_t dst = (j + 1) * N + i;
                edges_[index + j] = std::make_pair(src, dst);
            }
        }
    }

    std::vector<std::pair<int64_t, int64_t>> get_edges() { return edges_; }
};

bool is_sample_valid(int N, int num_neighbor, std::vector<int64_t> outputs,
                     std::vector<int64_t> output_counts)
{
    size_t index = 0;
    if (output_counts.size() != N) { return false; }
    for (int i = 0; i < N; i++) {
        int cnt = output_counts[i];
        std::unordered_set<int64_t> neighbors;
        for (int k = 0; k < num_neighbor; k++) {
            neighbors.insert((k + 1) * N + i);
        }
        for (int j = 0; j < cnt; j++) {
            int64_t neighbor = outputs[index++];
            if (neighbors.find(neighbor) == neighbors.end()) { return false; }
            neighbors.erase(neighbor);
        }
    }
    if (outputs.size() != index) { return false; }
    return true;
}

using V = int64_t;
using Quiver = quiver::quiver<V, quiver::CPU>;

void test_sample_kernel(int N, int neighbor, int k)
{
    auto g = simple_graph(N, neighbor);
    Quiver q(N, g.get_edges());
    std::vector<V> inputs(N);
    std::iota(inputs.begin(), inputs.end(), 0);

    auto res = q.sample_kernel(inputs, k);
    std::vector<V> outputs = std::get<0>(res);
    std::vector<V> output_counts = std::get<1>(res);
    bool ok = is_sample_valid(N, neighbor, outputs, output_counts);
    ASSERT_TRUE(ok);
}

TEST(test_quiver_cpu, test_1)
{
    test_sample_kernel(10, 5, 10);
    test_sample_kernel(100, 10, 5);
    test_sample_kernel(1000, 10, 10);
}
