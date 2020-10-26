#include <map>
#include <vector>

#include <quiver/quiver.cu.hpp>

#include "testing.hpp"

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

    bool operator==(const graph &g) const
    {
        if (n_ != g.n_) { return false; }
        if (m_ != g.m_) { return false; }
        if (!same_content(edges_, g.edges_)) { return false; }
        // TODO: test weight
        // if (adj_ != g.adj_) { return false; }
        return true;
    }
};

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

using V = int64_t;
using Quiver = quiver::quiver<V, quiver::CUDA>;

graph export_graph(const Quiver &q, bool reverse, bool &ok)
{
    std::vector<V> u;
    std::vector<V> v;
    q.get_edges(u, v);
    if (reverse) { std::swap(u, v); }
    graph g(q.size());
    const int m = u.size();
    for (int i = 0; i < m; ++i) {
        if (u[i] < v[i]) {
            ok &= g.add_edge(u[i], v[i]);
        } else if (u[i] == v[i]) {
            ok = false;
        }
    }
    return g;
}

void assert_graph_eq(const graph &g, const Quiver &q, bool reverse)
{
    bool ok = true;
    auto g1 = export_graph(q, reverse, ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ(g, g1);
}

void test_construct_1()
{
    auto g = gen_random_graph(100, 1000);

    std::vector<V> u(g.M());
    std::vector<V> v(g.M());
    g.get_edges(u, v);

    thrust::device_vector<V> row_idx = to_device<V>(u);
    thrust::device_vector<V> col_idx = to_device<V>(v);
    thrust::device_vector<V> edge_idx(g.M());
    thrust::sequence(edge_idx.begin(), edge_idx.end());
    Quiver q = Quiver::New(g.N(), row_idx, col_idx, edge_idx);

    printf("|V|=%d, |E|=%d\n", (int)q.size(), (int)q.edge_counts());
    assert_graph_eq(g, q, false);
    assert_graph_eq(g, q, true);
}

void test_construct_2()
{
    using W = float;
    auto g = gen_random_graph(100, 1000);

    std::vector<V> u(g.M());
    std::vector<V> v(g.M());
    std::vector<W> w(g.M());
    g.get_edges(u, v, w);

    thrust::device_vector<V> row_idx = to_device<V>(u);
    thrust::device_vector<V> col_idx = to_device<V>(v);
    thrust::device_vector<W> edge_weight = to_device<W>(w);
    thrust::device_vector<V> edge_idx(g.M());
    thrust::sequence(edge_idx.begin(), edge_idx.end());
    Quiver q = Quiver::New(g.N(), row_idx, col_idx, edge_idx, edge_weight);

    printf("|V|=%d, |E|=%d\n", (int)q.size(), (int)q.edge_counts());
    assert_graph_eq(g, q, false);
    assert_graph_eq(g, q, true);
}

TEST(test_quiver, test_1) { test_construct_1(); }

TEST(test_quiver, test_2) { test_construct_2(); }
