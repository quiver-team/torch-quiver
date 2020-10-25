#include <quiver/quiver.cu.hpp>

#include <map>
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
        if (i >= n_) { return false; }
        if (j >= n_) { return false; }

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

graph gen_random_graph(int n, int m)
{
    graph g(n);
    for (int i = 0; i < m; ++i) {
        constexpr int max_trials = 10;
        for (int j = 0; j < max_trials; ++j) {
            int x = rand() % n;
            int y = rand() % n;
            if (g.add_edge(x, y)) { break; }
        }
    }
    return g;
}

using V = int64_t;
using Quiver = quiver::quiver<V, quiver::CUDA>;

void test_construct_1()
{
    auto g = gen_random_graph(100, 1000);

    std::vector<V> u(g.M());
    std::vector<V> v(g.M());
    g.get_edges(u, v);

    thrust::device_vector<V> row_idx(g.M());
    thrust::device_vector<V> col_idx(g.M());
    thrust::device_vector<V> edge_idx(g.M());

    thrust::copy(u.begin(), u.end(), row_idx.begin());
    thrust::copy(v.begin(), v.end(), col_idx.begin());
    thrust::sequence(edge_idx.begin(), edge_idx.end());
    Quiver q = Quiver::New(g.N(), row_idx, col_idx, edge_idx);

    printf("|V|=%d, |E|=%d\n", (int)q.size(), (int)q.edge_counts());
}

void test_construct_2()
{
    using W = float;
    auto g = gen_random_graph(100, 1000);

    std::vector<V> u(g.M());
    std::vector<V> v(g.M());
    std::vector<W> w(g.M());
    g.get_edges(u, v, w);

    thrust::device_vector<V> row_idx(g.M());
    thrust::device_vector<V> col_idx(g.M());
    thrust::device_vector<V> edge_idx(g.M());
    thrust::device_vector<W> edge_weight(g.M());

    thrust::copy(u.begin(), u.end(), row_idx.begin());
    thrust::copy(v.begin(), v.end(), col_idx.begin());
    thrust::sequence(edge_idx.begin(), edge_idx.end());
    thrust::copy(w.begin(), w.end(), edge_weight.begin());
    Quiver q = Quiver::New(g.N(), row_idx, col_idx, edge_idx, edge_weight);

    printf("|V|=%d, |E|=%d\n", (int)q.size(), (int)q.edge_counts());
}

int main()
{
    test_construct_1();
    test_construct_2();
    return 0;
}
