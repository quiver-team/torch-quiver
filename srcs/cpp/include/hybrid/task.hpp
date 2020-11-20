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
    virtual void dispatch(std::vector<HeteroWorker> workers) = 0;
    virtual bool pull(HeteroWorker worker) = 0;
};

}  // namespace hybrid
