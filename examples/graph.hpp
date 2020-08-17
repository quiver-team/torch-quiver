#pragma once
#include <random>
#include <vector>

class graph
{
    using list = std::vector<int>;

    std::vector<list> neighbors_;

  public:
    graph(int n) : neighbors_(n) {}

    void add_edge(int i, int j) { neighbors_.at(i).push_back(j); }

    // int size() const;

    // int edge_count() const;

    // int degree(int i) const;

    // const list &neighbors(int i) const;

    // void simplify();

    // void debug() const;

    std::vector<std::pair<int, int>> edge_index()
    {
        std::vector<std::pair<int, int>> e;
        const int n = neighbors_.size();
        for (int i = 0; i < n; ++i) {
            for (auto j : neighbors_[i]) { e.push_back(std::make_pair(i, j)); }
        }
        return e;
    }

    static graph fast_gnp_random_graph(int n, double p)
    {
        graph g(n);
        const double lp = std::log(1.0 - p);
        int w = -1;
        int v = 1;
        while (v < n) {
            const double lr =
                std::log(1.0 - std::rand() / static_cast<double>(RAND_MAX));
            w = w + 1 + (int)(lr / lp);
            while (w >= v && v < n) {
                w = w - v;
                v = v + 1;
            }
            if (v < n) {
                g.add_edge(v, w);
                g.add_edge(w, v);
            }
        }
        // g.simplify();
        return g;
    }
};
