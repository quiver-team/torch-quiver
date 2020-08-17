#include <iostream>
#include <random>
#include <vector>

#include <quiver/quiver.hpp>
// #include <quiver/trace.hpp>
#include <tracer/stack>

#include "graph.hpp"

DEFINE_TRACE_CONTEXTS;

static std::random_device device;
static std::mt19937 g(device());

void check_eq(int x, int y)
{
    if (x != y) { throw std::runtime_error("check failed"); }
}

void bench_sample(const quiver::Quiver &q, int batch_size, int k,
                  int times = 100)
{
    fprintf(stderr,
            "sampling from G(%d, %d) with bs=%d, k=%d on %s for %d times\n",
            (int)q.size(), (int)q.edge_counts(), batch_size, k,
            device_name(q.device()), times);
    TRACE_SCOPE(__func__);
    std::vector<int> vertices(q.size());
    std::vector<int> batch(batch_size);
    std::iota(vertices.begin(), vertices.end(), 0);

    for (int i = 0; i < times; ++i) {
        std::sample(vertices.begin(), vertices.end(), batch.begin(),
                    batch.size(), g);
        {
            TRACE_SCOPE("q.sample");
            q.sample(batch, k);
        }
    }
}

void bench_sample(const quiver::Quiver &q)
{
    const auto batch_sizes = {
        1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16, 1 << 17,
    };
    const auto sample_sizes = {
        5,
        10,
        15,
    };
    const int times = 1;
    for (auto &bs : batch_sizes) {
        TRACE_SCOPE("bs=" + std::to_string(bs));
        for (auto &k : sample_sizes) {
            TRACE_SCOPE("k=" + std::to_string(k));
            bench_sample(q, std::min(bs, (int)q.size()), k, times);
        }
    }
}

template <typename T>
void read_vector(std::vector<T> &x, FILE *fp)
{
    check_eq(fread(x.data(), sizeof(T), x.size(), fp), x.size());
}

auto load_ogb(const std::string path)
{
    TRACE_SCOPE(__func__);
    std::vector<int64_t> col;
    std::vector<int64_t> rowptr;
    std::vector<int64_t> rowcount;
    {
        FILE *fp = fopen((path + "/col.data").c_str(), "rb");
        col.resize(123718280);
        read_vector(col, fp);
        fclose(fp);
    }
    {
        FILE *fp = fopen((path + "/rowcount.data").c_str(), "rb");
        rowcount.resize(2449029);
        read_vector(rowcount, fp);
        fclose(fp);
    }
    std::vector<std::pair<int, int>> edge_index;
    edge_index.reserve(col.size());
    const int n = rowcount.size();
    int offset = 0;
    for (int i = 0; i < n; ++i) {
        const int c = rowcount[i];
        for (int j = 0; j < c; ++j) {
            edge_index.push_back(std::make_pair(i, col[offset]));
            ++offset;
        }
    }
    return std::make_pair(2449029, edge_index);
}

void bench()
{
    int n = 100;
    auto g = graph::fast_gnp_random_graph(n, 0.01);
    for (auto &dev : {quiver::CUDA, quiver::CPU}) {
        printf("begin benchmark on device %s\n", quiver::device_name(dev));
        const auto edge_index = g.edge_index();
        auto q = quiver::Quiver::from_edge_index(n, std::move(edge_index), dev);
        bench_sample(*q);
    }
}

int main()
{
    const auto path = std::string(getenv("HOME")) + "/var/data/graph";
    const auto [n, edge_index] = load_ogb(path);
    for (auto &dev : {quiver::CUDA, quiver::CPU}) {
        auto q = [&] {
            TRACE_SCOPE("from_edge_index::" + std::string(device_name(dev)));
            return quiver::Quiver::from_edge_index(n, edge_index, dev);
        }();
        printf("quiver created\n");
        if (q) {
            TRACE_SCOPE(device_name(dev));
            bench_sample(*q);
        }
    }
    return 0;
}
