#pragma once
#include <torch/extension.h>

__device__ int find(const int64_t* offsets, const int device_count, const int64_t index){
    if(index < offsets[0]){
        return 0;
    }
    int i = 1;
    for(i = 1; i < device_count; i++){
        if(index < offsets[i]){
            return i - 1;
        }
    }
    return device_count - 1;
}
__global__ void quiver_tensor_gather(int64_t** dev_ptrs, const int64_t* offsets, const int device_count,
                                     const int64_t* indices, int indice_length, 
                                     const float* res,
                                     const int stride){

    // 
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // decide step
    unsigned int step = gridDim.x * blockDim.x;
    unsigned int start = tid;
    int64_t dev_index = 0;
    int64_t dev_offset = 0; 
    int64_t* dev_ptr = nullptr;
    int64_t src_copy_start = 0;
    int64_t dst_copy_start = 0;
    unsigned int copy_count = 0;

    while(start < indice_length){
        dev_index = find(offsets, device_count, indices[start]);
        dev_ptr = dev_ptrs[dev_index];
        dev_offset = indices[start] - offsets[dev_index];
        src_copy_start = dev_offset * stride;
        dst_copy_start = start * stride;
        for(copy_count = 0; copy_count < stride; copy_count ++){
            res[dst_copy_start + copy_count] = dev_ptr[src_copy_start + copy_count];
        }
        start += step;
    }
}