#pragma once
#include <iostream>
#include <memory>
#include <mutex>

namespace quiver
{
class Logger
{
    std::unique_ptr<std::lock_guard<std::mutex>> lk_;

  public:
    Logger(std::mutex &mu) : lk_(new std::lock_guard<std::mutex>(mu)) {}

    ~Logger() { std::cerr << std::endl; }

    template <typename M>
    Logger &operator<<(const M &msg)
    {
        std::cerr << msg;
        return *this;
    }
};

Logger DBG();
}  // namespace quiver
