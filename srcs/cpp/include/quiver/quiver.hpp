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

// TODO: do we need a unified interface for CPU and GPU implementation?
template <typename T, device_t device>
class quiver;
}  // namespace quiver
