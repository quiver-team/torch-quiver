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

  public:
    layer_sample_task(int scale, int fanout,
                      const std::vector<int> &all_num_seeds,
                      const std::vector<HeteroAddress> &all_src,
                      const std::vector<HeteroAddress> &all_dst)
        : scale_(scale),
          fanout_(fanout),
          all_num_seeds_(all_num_seeds),
          all_src_(all_src),
          all_dst_(all_dst)
    {
    }
};
}  // namespace sample
}  // namespace hybrid