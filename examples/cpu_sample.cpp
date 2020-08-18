#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <quiver/trace.hpp>

DEFINE_TRACE;

std::vector<int64_t> col;
std::vector<int64_t> rowptr;
std::vector<int64_t> rowcount;
std::vector<int64_t> rowcount_1;

void check_eq(int x, int y)
{
    if (x != y) { throw std::runtime_error("check failed"); }
}

template <typename T>
void read_vector(std::vector<T> &x, FILE *fp)
{
    check_eq(fread(x.data(), sizeof(T), x.size(), fp), x.size());
}

void read_lines(FILE *fp, std::vector<int64_t> &a)
{
    char line[32];
    while (fgets(line, 31, fp)) {
        int64_t x;
        sscanf(line, "%ld", &x);
        a.push_back(x);
    }
    printf("got %lu items\n", a.size());
}

void read_data(const std::string path)
{
    TRACE(__func__);
    {
        FILE *fp = fopen((path + "/col.txt").c_str(), "r");
        col.reserve(123718280);
        read_lines(fp, col);
        fclose(fp);
    }
    {
        FILE *fp = fopen((path + "/rowcount.txt").c_str(), "r");
        rowcount.reserve(2449029);
        read_lines(fp, rowcount);
        fclose(fp);
    }
    {
        FILE *fp = fopen((path + "/rowptr.txt").c_str(), "r");
        rowptr.reserve(2449030);
        read_lines(fp, rowptr);
        fclose(fp);
    }
}

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

void save_graph(const std::string path)
{
    TRACE(__func__);
    {
        FILE *fp = fopen((path + "/col.data").c_str(), "wb");
        fwrite(col.data(), sizeof(int64_t), col.size(), fp);
        fclose(fp);
    }
    {
        FILE *fp = fopen((path + "/rowcount.data").c_str(), "wb");
        fwrite(rowcount.data(), sizeof(int64_t), rowcount.size(), fp);
        fclose(fp);
    }
    {
        FILE *fp = fopen((path + "/rowptr.data").c_str(), "wb");
        fwrite(rowptr.data(), sizeof(int64_t), rowptr.size(), fp);
        fclose(fp);
    }
}

int main()
{
    const auto path = std::string(getenv("HOME")) + "/var/data/graph";
    // read_data(path);
    // save_graph(path);
    read_raw(path);
    rowcount_1.resize(rowptr.size());
    std::adjacent_difference(rowptr.begin(), rowptr.end(), rowcount_1.begin());
    rowcount_1.erase(rowcount_1.begin());
    std::cout << std::boolalpha
              << std::equal(rowcount.begin(), rowcount.end(),
                            rowcount_1.begin())
              << std::endl;
    return 0;
}
