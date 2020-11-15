#pragma once
#include <algorithm>
#include <map>
#include <tuple>
#include <vector>

template <size_t i>
struct std_get {
    template <typename T>
    auto operator()(const T &t) const
    {
        return std::get<i>(t);
    }
};

class graph
{
    const size_t n_;
    size_t m_;

    std::vector<std::map<int, float>> adj_;
    std::vector<std::pair<int, int>> edges_;

  public:
    graph(int n) : n_(n), adj_(n), m_(0) {}

    bool add_edge(int i, int j, float w = 0.0)
    {
        if (i == j) { return false; }
        if (adj_[i].count(j) > 0) { return false; }
        if (i < 0 || i >= n_) { return false; }
        if (j < 0 || j >= n_) { return false; }

        adj_[i][j] = w;
        adj_[j][i] = w;
        edges_.push_back(std::make_pair(i, j));
        edges_.push_back(std::make_pair(j, i));
        return true;
    }

    size_t N() const { return n_; }
    size_t M() const { return edges_.size(); }

    template <typename V>
    void get_edges(std::vector<V> &u, std::vector<V> &v)
    {
        u.resize(edges_.size());
        v.resize(edges_.size());

        std::transform(edges_.begin(), edges_.end(), u.begin(), std_get<0>());
        std::transform(edges_.begin(), edges_.end(), v.begin(), std_get<1>());
    }

    template <typename V, typename W>
    void get_edges(std::vector<V> &u, std::vector<V> &v, std::vector<W> &w)
    {
        get_edges(u, v);
        const int m = edges_.size();
        w.resize(m);
        for (int i = 0; i < m; ++i) {
            V x = u[i];
            V y = v[i];
            w[i] = adj_.at(x).at(y);
        }
    }
};

graph gen_random_graph(int n, int m);

void std_sample_i64(const int64_t *in, size_t total, int64_t *out,
                    size_t count);
