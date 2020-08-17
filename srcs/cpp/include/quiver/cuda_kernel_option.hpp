#pragma once

class cuda_kernel_option
{
    size_t threads_per_block_;
    size_t max_block_count_;

  public:
    cuda_kernel_option() : threads_per_block_(16), max_block_count_(64) {}
};
