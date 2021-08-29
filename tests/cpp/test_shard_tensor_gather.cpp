#include "./shard_tensor.cu.hpp"
#include <thrust/device_vector.h>
#include <torch/extension.h>
#include <stdio.h>
#include <stdlib.h>

int main(){
    int numGPUs, numElems =  40000;
    cudaGetDeviceCount(&numGPUs);
    std::vector<float *> buffers(numGPUs);
    std::vector<int64_t> offset_host;
    offset_host.push_back(0);
    offset_host.push_back(numElems);

    std::vector<int64_t> indices_host;
    for(int index = 0; index < numElems; index++){
        indices_host.push_back(rand() % (numElems * 2));
    }

    int64_t* offset_device;
    cudaMalloc((void**) &offset_device, sizeof(int64_t) * offset_host.size());
    int64_t* indices_device;
    cudaMalloc((void**) &indices_device, sizeof(int64_t) * indices_host.size());


    cudaMemcpy(offset_device, &offset_host, sizeof(int64_t) * offset_host.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(indices_device, &indices_host, sizeof(int64_t) * indices_host.size(), cudaMemcpyHostToDevice);

    float* res_device;
    cudaMalloc((void**) &res_device, sizeof(float) * numElems);

    for (int d = 0; d < numGPUs; d++) {
        cudaSetDevice(d);
        cudaMalloc(&buffers[d], numElems * sizeof(float));
        cudaMemset(buffers[d], 0, numElems * sizeof(float));
        cudaCheckError();
    }
    /*
    __global__ void quiver_tensor_gather(float** dev_ptrs, const int64_t* offsets, const int device_count,
                                     const int64_t* indices, int indice_length, 
                                     float* res,
                                     const int stride){
    */

    quiver_tensor_gather<<<1024, 1024>>>(&buffers[0], offset_device, 2, indices_device, numElems, res_device, 1);
    cudaDeviceSynchronize();

}