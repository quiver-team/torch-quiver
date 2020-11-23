#include <hybrid/table.hpp>

namespace hybrid
{
bool TaskTable::add_flow(TaskFlow flow)
{
    std::lock_guard _lk(mu_);
    if (capacity_ - available_ < flow.size()) { return false; }
    auto nodes = flow.get_nodes();
    auto slots = allocate(nodes.size());
    for (int i = 0; i < nodes.size(); i++) {
        int index = slots[i];
        std::vector<int> children;
        for (auto child : nodes[i]->children) {
            for (int j = 0; j < nodes.size(); j++) {
                if (child == nodes[j]) { children.push_back(slots[j]); }
            }
        }
        entries_[index] = TaskEntry(index, nodes[i]->prev, std::move(children),
                                    nodes[i]->task);
    }
}

void TaskTable::pull(HeteroWorker worker)
{
    mu_.lock();
    for (int i = 0; i < capacity_; i++) {
        if (entries_[i].pull(worker, &mu_)) {
            mu_.lock();
            if (entries_[i].finished()) {
                available_++;
                used_--;
                std::vector<int> children = entries_[i].dependent_tasks();
                for (int child : children) { entries_[child].prepare(); }
            }
            mu_.unlock();
            return;
        }
    }
    mu_.unlock();
}

std::vector<int> TaskTable::allocate(int size)
{
    std::vector<int> alloc(size);
    for (int i = 0; i < capacity_; i++) {
        if (entries_[i].available()) {
            alloc[--size] = i;
            if (size < 0) {
                used_ += size;
                available_ -= size;
                return std::move(alloc);
            }
        }
    }
}
}  // namespace hybrid
