#pragma once
#include <vector>

#include <hybrid/hetero.hpp>
#include <hybrid/task.hpp>

namespace hybrid
{
namespace sample
{
class layer_sample_runner : public TaskRunner
{
    int fanout_;
    int num_seeds_;
    HeteroAddress src_;
    HeteroAddress dst_;

  public:
    layer_sample_runner(HeteroWorker worker, int fanout, int num_seeds,
                        HeteroAddress src, HeteroAddress dst)
        : TaskRunner(worker),
          fanout_(fanout),
          num_seeds_(num_seeds_),
          src_(src),
          dst_(dst)
    {
    }
    void run();
};
class layer_sample_task : public Task
{
    int scale_;
    int fanout_;
    std::vector<HeteroAddress> all_src_;
    std::vector<HeteroAddress> all_dst_;
    std::vector<int> all_num_seeds_;
    std::vector<TaskRunner> all_runner_;

  public:
    layer_sample_task(int scale, int fanout, std::vector<int> all_num_seeds,
                      std::vector<HeteroAddress> all_src,
                      std::vector<HeteroAddress> all_dst)
        : scale_(scale),
          fanout_(fanout),
          all_num_seeds_(std::move(all_num_seeds)),
          all_src_(std::move(all_src)),
          all_dst_(std::move(all_dst))
    {
    }
    void dispatch(std::vector<HeteroWorker> workers);
    bool pull(HeteroWorker worker);
};
}  // namespace sample
}  // namespace hybrid
