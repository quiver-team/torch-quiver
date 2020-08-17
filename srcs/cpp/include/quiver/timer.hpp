#pragma once
#include <chrono>
#include <string>

namespace quiver
{
class timer
{
    using clock_t = std::chrono::high_resolution_clock;
    using instant_t = std::chrono::time_point<clock_t>;
    using duration_t = std::chrono::duration<double>;
    const std::string name_;
    const instant_t t0_;

  public:
    timer(std::string name) : name_(std::move(name)), t0_(clock_t::now()) {}

    ~timer()
    {
        const duration_t d = clock_t::now() - t0_;
        using namespace std::literals::chrono_literals;
        if (d > 1s) {
            printf("%s took %.2fs\n", name_.c_str(), d.count());
        } else {
            printf("%s took %.2fms\n", name_.c_str(), 1e3 * d.count());
        }
    }
};
}  // namespace quiver
