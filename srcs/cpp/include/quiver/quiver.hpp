#pragma once
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace quiver
{
enum device_t {
    CPU,
    CUDA,
};

inline const char *device_name(device_t device)
{
    switch (device) {
    case CPU:
        return "CPU";
    case CUDA:
        return "CUDA";
    default:
        throw std::invalid_argument("invalid device");
    }
}

class Quiver
{
    static std::unique_ptr<Quiver>
    from_edge_index_cpu(int n, std::vector<std::pair<int, int>> edge_index);

    static std::unique_ptr<Quiver>
    from_edge_index_cuda(int n, std::vector<std::pair<int, int>> edge_index);

  public:
    Quiver();
    virtual ~Quiver() = default;

    virtual void sample(const std::vector<int> &vertices, int k) const
    {
        throw std::runtime_error("sample not implemented.");
    }

    virtual size_t size() const = 0;
    virtual size_t edge_counts() const = 0;
    virtual device_t device() const = 0;

    static std::unique_ptr<Quiver>
    from_edge_index(int n, std::vector<std::pair<int, int>> edge_index,
                    device_t device);
};

template <typename T, device_t device>
class quiver;
}  // namespace quiver
