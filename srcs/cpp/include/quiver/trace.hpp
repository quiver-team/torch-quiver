#pragma once
#include <chrono>

namespace quiver
{
class tracer
{
    using clock_t = std::chrono::high_resolution_clock;
    using instant_t = std::chrono::time_point<clock_t>;

    const std::string name_;
    const instant_t t0_;

  public:
    tracer(const std::string name) : name_(std::move(name)), t0_(clock_t::now())
    {
        // printf("%*s{ // %s\n", indent * tab, "", name_);
        // ++indent;
    }
    ~tracer()
    {
        using duration_t = std::chrono::duration<double>;
        // --indent;
        // printf("%*s} // %s\n", indent * tab, "", name_);
        const duration_t d = clock_t::now() - t0_;
        printf("%s took %fms\n", name_.c_str(), d.count() * 1000);
    }
};

}  // namespace quiver

#define TRACE(e) ::quiver::tracer __(e);
