#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <quiver/trace.hpp>

void check_eq(int x, int y)
{
    if (x != y) { throw std::runtime_error("check failed"); }
}

template <typename T>
void read_vector(std::vector<T> &x, FILE *fp)
{
    check_eq(fread(x.data(), sizeof(T), x.size(), fp), x.size());
}

std::vector<int64_t> col;
std::vector<int64_t> rowptr;
std::vector<int64_t> rowcount;

void read_raw(const std::string path)
{
    TRACE(__func__);
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
    {
        FILE *fp = fopen((path + "/rowptr.data").c_str(), "rb");
        rowptr.resize(2449030);
        read_vector(rowptr, fp);
        fclose(fp);
    }
}

int main()
{
    const auto path = std::string(getenv("HOME")) + "/var/data/graph";
    // read_data(path);
    // save_graph(path);
    read_raw(path);
    return 0;
}
