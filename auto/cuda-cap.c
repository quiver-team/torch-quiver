#include <cuda.h>
#include <stdio.h>

#define CHECK(e)                                                               \
    {                                                                          \
        CUresult result = e;                                                   \
        if (result != CUDA_SUCCESS) {                                          \
            fprintf(stderr, "%s failed\n", #e);                                \
            exit(1);                                                           \
        }                                                                      \
    }

void check_cuda_device_comp_cap(int device)
{
    int comp_cap_major = 0;
    int comp_cap_minor = 0;
    CHECK(cuDeviceGetAttribute(
        &comp_cap_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));

    CHECK(cuDeviceGetAttribute(
        &comp_cap_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));

    printf("%d.%d\n", comp_cap_major, comp_cap_minor);
}

int main()
{
    int flags = 0;
    CHECK(cuInit(flags));

    // int driver_version;
    // CHECK(cuDriverGetVersion(&driver_version));
    // printf("driver version: %d\n", driver_version);

    int device_count;
    CHECK(cuDeviceGetCount(&device_count));
    // printf("device count: %d\n", device_count);

    for (int device = 0; device < device_count; ++device) {
        check_cuda_device_comp_cap(device);
        break;
    }
    return 0;
}
