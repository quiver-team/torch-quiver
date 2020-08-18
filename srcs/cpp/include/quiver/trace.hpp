#pragma once
#include <chrono>
#include <cstdio>
#include <string>

bool trace_enable();

extern bool _trace_enable;

namespace quiver
{
class tracer
{
    using clock_t = std::chrono::high_resolution_clock;
    using instant_t = std::chrono::time_point<clock_t>;

    const std::string name_;
    const instant_t t0_;

    static constexpr int tab = 4;

    static int indent;

  public:
    tracer(const std::string name) : name_(std::move(name)), t0_(clock_t::now())
    {
        if (_trace_enable) {
            printf("%*s{ // %s\n", indent * tab, "", name_.c_str());
            ++indent;
        }
    }

    ~tracer()
    {
        if (_trace_enable) {
            using duration_t = std::chrono::duration<double>;
            --indent;
            const duration_t d = clock_t::now() - t0_;
            printf("%*s} // %s took %fms\n", indent * tab, "", name_.c_str(),
                   d.count() * 1000);
        }
    }
};
}  // namespace quiver

#define TRACE(e) ::quiver::tracer __(e);

#define DEFINE_TRACE                                                           \
    int ::quiver::tracer::indent = 0;                                          \
                                                                               \
    bool trace_enable()                                                        \
    {                                                                          \
        if (getenv("ENABLE_TRACE")) { return true; }                           \
        return false;                                                          \
    }                                                                          \
                                                                               \
    bool _trace_enable = trace_enable();
