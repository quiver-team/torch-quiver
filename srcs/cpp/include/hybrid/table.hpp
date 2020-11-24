#pragma once
#include <atomic>
#include <cassert>
#include <mutex>
#include <vector>

#include <hybrid/hetero.hpp>
#include <hybrid/task.hpp>

namespace hybrid
{
struct TaskNode {
    Task *task;
    int prev;
    std::vector<TaskNode *> children;

    TaskNode(Task *task, int prev, std::vector<TaskNode *> children)
        : task(task), prev(prev), children(std::move(children))
    {
    }
};

class TaskFlow
{
    std::vector<TaskNode *> nodes_;

  public:
    TaskFlow(std::vector<TaskNode *> nodes) : nodes_(std::move(nodes)) {}

    size_t size() { return nodes_.size(); }

    std::vector<TaskNode *> get_nodes() { return std::move(nodes_); }
};

struct TaskMeta {
    int id;
    int prev;
    std::vector<int> children;
    bool valid;

    TaskMeta() : valid(false) {}

    TaskMeta(int id, int prev, std::vector<int> children)
        : id(id), prev(prev), children(std::move(children)), valid(true)
    {
    }
};

class TaskEntry
{
    TaskMeta meta_;
    Task *task_;

  public:
    TaskEntry() : meta_(), task_(nullptr) {}

    TaskEntry(int id, int prev, std::vector<int> children, Task *task)
        : meta_(id, prev, children), task_(task)
    {
    }

    void prepare()
    {
        assert(meta_.prev > 0);
        meta_.prev--;
    }

    bool available() { return !meta_.valid || task_->finished(); }

    bool ready() { return meta_.prev == 0; }

    bool finished() { return task_->finished(); }

    bool valid() { return meta_.valid; }

    bool pull(HeteroWorker worker, std::mutex *mu)
    {
        if (!ready()) { return false; }
        if (!task_->can_pull(worker)) { return false; }
        mu->unlock();
        task_->pull(worker);
        return true;
    }

    std::vector<int> dependent_tasks() { return std::move(meta_.children); }
};

class TaskTable
{
    std::mutex mu_;
    std::vector<TaskEntry> entries_;
    int capacity_;
    int available_;
    int used_;

    std::vector<int> allocate(int size);

  public:
    TaskTable(int cap)
        : capacity_(cap), available_(cap), used_(0), entries_(cap)
    {
    }

    bool add_flow(TaskFlow flow);

    void pull(HeteroWorker worker);
};
}  // namespace hybrid
