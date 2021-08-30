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


__global__ void delay(volatile int *flag,
    unsigned long long timeout_clocks = 10000000) {
    // Wait until the application notifies us that it has completed queuing up the
    // experiment, or timeout and exit, allowing the application to make progress
    long long int start_clock, sample_clock;
    start_clock = clock64();

    while (!*flag) {
        sample_clock = clock64();

        if (sample_clock - start_clock > timeout_clocks) {
            break;
        }
    }
}
__device__ int find(const int64_t* offsets, const int device_count, const int64_t index){
    int i = 1;
    for(i = 1; i < device_count; i++){
        if(index < offsets[i]){
            return i - 1;
        }
    }
    return device_count - 1;
}
__global__ void quiver_tensor_gather(float** dev_ptrs, const int64_t* offsets, const int device_count,
                                     const int64_t* indices, int indice_length, 
                                     float* res,
                                     const int stride){

    // 
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    unsigned int start = tid;
    int64_t dev_index = 0;
    int64_t dev_offset = 0; 
    float* dev_ptr;
    int64_t src_copy_start = 0;
    int64_t dst_copy_start = 0;
    unsigned int copy_count = 0;
    if(tid == 0){
    	printf("check tid = %d, start = %d, indices_length = %d \n", tid, start, indice_length);
	    printf("check offset[1] = %d, indices[10] = %d, dev_ptrs[0][1] = %d \n", offsets[1], indices[10], dev_ptrs[0][1]);

        dev_index = find(offsets, device_count, 49000);

        dev_offset = 90000 - offsets[dev_index];
        printf("index = %lld, dev_index = %lld, dev_offset = %lld \n", indices[start], dev_index, dev_offset);
    }
    __syncthreads();
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
int main(){
    int numGPUs, numElems =  40000;
    cudaGetDeviceCount(&numGPUs);
    std::cout<<"device count = " << numGPUs <<std::endl;
    std::vector<float *> buffers(numGPUs);
    std::vector<int64_t> offset_host;
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

    float* res_device;
    float* res_host = (float*) malloc(sizeof(float) * numElems);
    cudaMalloc((void**) &res_device, sizeof(float) * numElems);
    cudaCheckError();


    for (int d = 0; d < numGPUs; d++) {
         cudaSetDevice(d);
         cudaMalloc((void**) &buffers[d], numElems * sizeof(float));
         cudaMemset(buffers[d], 0, numElems * sizeof(float));
         cudaCheckError();
    }


    cudaSetDevice(0);
    float ** buffers_device;
    cudaMalloc((void ***) &buffers_device, sizeof(float*) * numGPUs);
    cudaMemcpy(buffers_device, &buffers[0], sizeof(float*) * buffers.size(), cudaMemcpyHostToDevice);
    cudaCheckError();
    
    std::cout<<"all data initialization finished " <<std::endl;


    quiver_tensor_gather<<<1024, 512>>>(buffers_device, offset_device, numGPUs, indices_device, numElems, res_device, 1);
    cudaDeviceSynchronize();
    cudaDeviceSynchronize();
    cudaCheckError();

    std::cout<<"test finished " <<std::endl;
}
