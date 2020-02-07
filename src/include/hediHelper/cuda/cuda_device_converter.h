#include <cuda_runtime.h>

void *_device(void *object, int size) {
    void *devicePtr;
    cudaMalloc(&devicePtr, size);
    cudaMemcpy(devicePtr, &object, size, cudaMemcpyHostToDevice);
    return devicePtr;
}
