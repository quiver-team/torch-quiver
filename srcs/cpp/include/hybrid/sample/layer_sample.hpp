#pragma once
#include <atomic>
#include <memory>
#include <vector>

#include <hybrid/hetero.hpp>
#include <hybrid/task.hpp>

namespace hybrid
{
namespace sample
{
enum layer_sample_state {
    PREPARE,
    READY,
    RUN,
    DONE,
};

class layer_sample_runner : public TaskRunner
{
    int fanout_;
    int num_seeds_;
    HeteroAddress src_;
    HeteroAddress dst_;
    HeteroAddress cnt_;
    std::atomic<int> state_;

  public:
    layer_sample_runner(HeteroWorker worker, int fanout, int num_seeds,
                        HeteroAddress src, HeteroAddress dst, HeteroAddress cnt)
        : TaskRunner(worker),
          fanout_(fanout),
          num_seeds_(num_seeds_),
          src_(src),
          dst_(dst),
          cnt_(cnt),
          state_(PREPARE)
    {
    }

    void set_state(layer_sample_state state)
    {
        state_ = static_cast<int>(state);
    }

    layer_sample_state get_state()
    {
        return static_cast<layer_sample_state>(state_.load());
    }

    void run();
};

class layer_sample_task : public Task
{
    int scale_;
    int fanout_;
    std::vector<HeteroAddress> all_src_;
    std::vector<HeteroAddress> all_dst_;
    std::vector<HeteroAddress> all_cnt_;
    std::vector<int> all_num_seeds_;
    std::vector<std::unique_ptr<TaskRunner>> all_runner_;

  public:
    layer_sample_task(int scale, int fanout, std::vector<int> all_num_seeds,
                      std::vector<HeteroAddress> all_src,
                      std::vector<HeteroAddress> all_dst,
                      std::vector<HeteroAddress> all_cnt)
        : scale_(scale),
          fanout_(fanout),
          all_num_seeds_(std::move(all_num_seeds)),
          all_src_(std::move(all_src)),
          all_dst_(std::move(all_dst)),
          all_cnt_(std::move(all_cnt))
    {
    }

    void dispatch(std::vector<HeteroWorker> workers);

    bool can_pull(HeteroWorker worker);

    void pull(HeteroWorker worker);

    bool finished();
};
}  // namespace sample
}  // namespace hybrid
