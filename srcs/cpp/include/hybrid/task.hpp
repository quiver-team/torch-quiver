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

    bool set_worker(HeteroWorker worker)
    {
        if (worker_.can_fit(worker)) {
            worker_ = worker;
            return true;
        }
        return false;
    }

    virtual void run() = 0;
};

class Task
{
  public:
    Task() {}

    virtual void dispatch(std::vector<HeteroWorker> workers) = 0;

    virtual bool can_pull(HeteroWorker worker) = 0;

    virtual void pull(HeteroWorker worker) = 0;

    virtual bool finished() = 0;
};

}  // namespace hybrid
