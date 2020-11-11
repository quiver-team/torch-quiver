#include "common.hpp"

#include <algorithm>
#include <random>

graph gen_random_graph(int n, int m)
{
    graph g(n);
    for (int i = 0; i < m; ++i) {
        constexpr int max_trials = 10;
        for (int j = 0; j < max_trials; ++j) {
            int x = rand() % n;
            int y = rand() % n;
            if (g.add_edge(x, y, i + 1)) { break; }
        }
    }
    return g;
}

void std_sample_i64(const int64_t *in, size_t total, int64_t *out, size_t count)
{
    static std::random_device rd;
    static std::mt19937 rg(rd());
    std::sample(in, in + total, out, count, rg);
}
