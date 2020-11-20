#pragma once
#include <vector>

#include <hybrid/hetero.hpp>

namespace hybrid
{
class TaskRunner
{
    HeteroWorker worker_;

  public:
    TaskRunner(HeteroWorker worker) : worker_(worker) {}
    HeteroWorker get_worker() { return worker_; }
    virtual void run() = 0;
};
class Task
{
  public:
    Task() {}
    virtual void dispatch(const std::vector<HeteroWorker> &workers) = 0;
};

}  // namespace hybrid