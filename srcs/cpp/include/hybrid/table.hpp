#pragma once
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <hybrid/hetero.hpp>
#include <hybrid/task.hpp>

namespace hybrid
{
struct TaskNode {
    Task *task;
    int prev;
    std::vector<TaskNode *> children;
};

class TaskFlow
{
    std::vector<TaskNode *> nodes_;

  public:
    TaskFlow(std::vector<TaskNode *> nodes) : nodes_(std::move(nodes)) {}
    vector<TaskNode *> get_nodes() { return std::move(nodes_); }
};

struct TaskMeta {
    int64_t id;
    std::atomic<int> prev;
    std::vector<int64_t> children;
};

class TaskEntry
{
    TaskMeta meta_;
    Task *task_;

  public:
    bool ready() { return meta_.prev == 0; }

    bool finished() { return task_->finished(); }

    bool pull(HeteroWorker worker) {
        if (task_->can_pull(worker)) {
            return false;
        } else {
            task_->pull(worker);
            return true;
        }
    }

    std::vector<int64_t> dependent_tasks() {
        return meta_.children;
    }
};

class TaskTable
{
    std::mutex mu_;
    std::unordered_map<int64_t, TaskEntry> entries_;
    int capacity_;
    int finished_;
    int ready_;
    int size_;

  public:
    TaskTable(int cap) : capacity_(cap), finished_(0), ready_(0), size_(0) {}

    void add_flow(TaskFlow flow);

    void pull(HeteroWorker worker);
};
}  // namespace hybrid
