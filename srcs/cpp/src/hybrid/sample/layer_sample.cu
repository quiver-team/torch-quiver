#include <hybrid/sample/layer_sample.hpp>

namespace hybrid
{
namespace sample
{
void layer_sample_task::dispatch(std::vector<HeteroWorker> workers)
{
    assert(workers.size() == scale, "");

    for (int i = 0; i < scale_; i++) {
        all_runner_.push_back(layer_sample_runner(
            workers[i], fanout_, all_num_seeds_[i], all_src_[i], all_dst_[i]));
    }
}
bool layer_sample_task::pull(HeteroWorker worker)
{
    for (auto &runner : all_runner_) {
        if (runner.get_worker() == worker) {
            runner.run();
            return true;
        }
    }
    return false;
}
void layer_sample_runner::run() {}
}  // namespace sample
}  // namespace hybrid
