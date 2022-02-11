#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
// includes

#include <cuda.h>
#include <cassert>
#include <iostream>
#include <memory>
#include <iostream>
#include <vector>
#include <unistd.h>

#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }


#define WARP_SIZE 32

__device__ int find(const int64_t *offsets, const int device_count,
                    const int64_t index)
{
    int i = 1;
    for (i = 1; i < device_count; i++) {
        if (index < offsets[i]) { return i - 1; }
    }
    return device_count - 1;
}

__global__ void quiver_tensor_update(float **dev_ptrs, const int64_t *offsets,
                                     const int device_count,
                                     const int64_t *indices, int indice_length,
                                     float *update_data, const int stride,
                                     const int *access_book,
                                     const int ignore_access_book)
{
    //
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    // each warp take charge of one-feature copy
    unsigned int warp_id = tid / WARP_SIZE;
    unsigned int warp_step = step / WARP_SIZE;

    unsigned int warp_start = warp_id;
    unsigned int thread_start = tid % WARP_SIZE;

    int64_t dev_index = 0;
    int64_t dev_offset = 0;
    float *dev_ptr;
    int64_t src_copy_start = 0;
    int64_t dst_copy_start = 0;

    unsigned int local_start = thread_start;
    while (warp_start < indice_length) {
        local_start = thread_start;
        dev_index = find(offsets, device_count, indices[warp_start]);
        // we only copy data from reachable device
        if (ignore_access_book || access_book[dev_index] == 1) {
            dev_ptr = dev_ptrs[dev_index];
            dev_offset = indices[warp_start] - offsets[dev_index];
            src_copy_start = dev_offset * stride;
            dst_copy_start = warp_start * stride;
            for (; local_start < stride; local_start += WARP_SIZE) {
                dev_ptr[src_copy_start + local_start] = update_data[dst_copy_start + local_start];
            }
        }
        warp_start += warp_step;
    }

}

__global__ void quiver_tensor_gather(float **dev_ptrs, const int64_t *offsets,
                                     const int device_count,
                                     const int64_t *indices, int indice_length,
                                     float *res, const int stride,
                                     const int *access_book,
                                     const int ignore_access_book)
{

    //
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    // each warp take charge of one-feature copy
    unsigned int warp_id = tid / WARP_SIZE;
    unsigned int warp_step = step / WARP_SIZE;

    unsigned int warp_start = warp_id;
    unsigned int thread_start = tid % WARP_SIZE;

    int64_t dev_index = 0;
    int64_t dev_offset = 0;
    float *dev_ptr;
    int64_t src_copy_start = 0;
    int64_t dst_copy_start = 0;

    unsigned int local_start = thread_start;
    while (warp_start < indice_length) {
        local_start = thread_start;
        dev_index = find(offsets, device_count, indices[warp_start]);
        // we only copy data from reachable device
        if (ignore_access_book || access_book[dev_index] == 1) {
            dev_ptr = dev_ptrs[dev_index];
            dev_offset = indices[warp_start] - offsets[dev_index];
            src_copy_start = dev_offset * stride;
            dst_copy_start = warp_start * stride;
            for (; local_start < stride; local_start += WARP_SIZE) {
                res[dst_copy_start + local_start] =
                    dev_ptr[src_copy_start + local_start];
            }
        }
        warp_start += warp_step;
    }
}

__global__ void
quiver_tensor_gather_aligned(float **dev_ptrs, const int64_t *offsets,
                             const int device_count, const int64_t *indices,
                             int indice_length, float *res, const int stride)
{

    //
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int num_thread = gridDim.x * blockDim.x;

    unsigned int warp_start = thread_id;
    // unsigned int warp_end = (thread_id + 1) * WARP_SIZE;
    // unsigned int thread_local = thread_id % WARP_SIZE;

    int64_t dev_index = 0;
    int64_t dev_offset = 0;
    float *dev_ptr;
    int64_t src_copy_start = 0;
    int64_t dst_copy_start = 0;
    unsigned int output_index, output_offset;

    while (warp_start < indice_length * stride) {
        output_index = warp_start / stride;
        output_offset = warp_start % stride;
        dev_index = find(offsets, device_count, indices[output_index]);
        dev_ptr = dev_ptrs[dev_index];
        dev_offset = indices[output_index] - offsets[dev_index];

        src_copy_start = dev_offset * stride + output_offset;
        dst_copy_start = output_index * stride + output_offset;
        res[dst_copy_start] = dev_ptr[src_copy_start];
        warp_start += num_thread;
    }
}

void test_shardtensor_gather(){

}
int main(){
    int numGPUs, numElems =  40000;
    cudaGetDeviceCount(&numGPUs);
    int current_device = 0;
    std::cout<<"device count = " << numGPUs <<std::endl;
    std::vector<float *> buffers(numGPUs);
    std::vector<int64_t> offset_host;
    std::vector<int> access_book;
    std::vector<cudaStream_t> stream(numGPUs);


    std::cout<<"offset_host initialization finished " <<offset_host.size() <<std::endl;

    std::vector<int64_t> indices_host;
    int offset_val = 0;
    for(int index = 0; index < numElems; index++){
        indices_host.push_back(rand() % (numElems * numGPUs));
        offset_host.push_back(offset_val);
        offset_val += numElems;
    }

    std::cout<<"indices_host initialization finished " <<indices_host.size() <<std::endl;

    // P2P Initilization
    for (int i = 0; i < numGPUs; i++) {
        cudaSetDevice(i);
        for (int j = i + 1; j < numGPUs; j++) {
          int access = 0;
          cudaDeviceCanAccessPeer(&access, i, j);
          if (access) {
            cudaSetDevice(i);
            cudaDeviceEnablePeerAccess(j, 0);
            cudaCheckError();
            cudaSetDevice(j);
            cudaDeviceEnablePeerAccess(i, 0);
            cudaCheckError();
          }
        }
    }


    
    
    int64_t* offset_device;
    cudaMalloc((void**) &offset_device, sizeof(int64_t) * offset_host.size());
    cudaMemcpy(offset_device, &offset_host[0], sizeof(int64_t) * offset_host.size(), cudaMemcpyHostToDevice);
    cudaCheckError();

    int64_t* indices_device;
    cudaMalloc((void**) &indices_device, sizeof(int64_t) * indices_host.size());
    cudaMemcpy(indices_device, &indices_host[0], sizeof(int64_t) * indices_host.size(), cudaMemcpyHostToDevice);
    cudaCheckError();

    float* data_device;
    float* data_host = (float*) malloc(sizeof(float) * numElems);
    // randomly initialize data 
    for(int index = 0; index < numElems; index++){
        data_host[index] = rand() % (numElems * numGPUs);
    }

    cudaMalloc((void**) &data_device, sizeof(float) * numElems);
    cudaCheckError();


    for (int d = 0; d < numGPUs; d++) {
         cudaSetDevice(d);
         cudaMalloc((void**) &buffers[d], numElems * sizeof(float));
         cudaMemset(buffers[d], 0, numElems * sizeof(float));
         cudaCheckError();
    }


    cudaSetDevice(current_device);
    for (int i = 0; i < numGPUs; i++) {
        int access = 0;
        cudaDeviceCanAccessPeer(&access, current_device, i);
        if(access || i ==  current_device){
            access_book.push_back(1);
        }else{
            access_book.push_back(0);
        }
    }


    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, (void*) buffers[1]);
    
    cudaPointerGetAttributes(&attributes,  attributes.devicePointer);


    float ** buffers_device;
    cudaMalloc((void ***) &buffers_device, sizeof(float*) * numGPUs);
    cudaMemcpy(buffers_device, &buffers[0], sizeof(float*) * buffers.size(), cudaMemcpyHostToDevice);
    cudaCheckError();
    
    int* access_book_device;
    cudaMalloc((void **) &access_book_device, sizeof(int) * access_book.size());
    cudaMemcpy(buffers_device, &buffers[0], sizeof(float*) * buffers.size(), cudaMemcpyHostToDevice);
    cudaCheckError();
    

    std::cout<<"all data initialization finished " <<std::endl;

    // Uncomment this if you want to test quiver_tensor_gather
    //quiver_tensor_gather<<<1024, 512>>>(buffers_device, offset_device, numGPUs, indices_device, numElems, data_device, 1, access_book_device, 1);
    quiver_tensor_update<<<1024, 512>>>(buffers_device, offset_device, numGPUs, indices_device, numElems, data_device, 1, access_book_device, 1);
    cudaDeviceSynchronize();
    cudaCheckError();

    std::cout<<"test finished " <<std::endl;
}
