#include "sparseDataStruct/helper/vector_helper.h"
#include "sparseDataStruct/vector_dense.hpp"

__host__ D_Array::D_Array(int n, bool isDevice)
    : n(n), isDevice(isDevice) {
    MemAlloc();
}

__host__ D_Array::D_Array(const D_Array &m, bool copyToOtherMem)
    : D_Array(m.n, m.isDevice ^ copyToOtherMem) {
    cudaMemcpyKind memCpy =
        (m.isDevice)
            ? (isDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost
            : (isDevice) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    gpuErrchk(cudaMemcpy(vals, m.vals, sizeof(T) * n, memCpy));
}

__host__ void D_Array::MemAlloc() {
    if (isDevice) {
        gpuErrchk(cudaMalloc(&vals, n * sizeof(T)));
        gpuErrchk(cudaMalloc(&_device, sizeof(D_Array)));
        gpuErrchk(cudaMemcpy(_device, this, sizeof(D_Array),
                             cudaMemcpyHostToDevice));
    } else {
        vals = new T[n];
    }
}

__host__ __device__ void D_Array::Print() {
    printf("[ ");
#ifndef __CUDA_ARCH__
    if (isDevice) {
        printVector<<<1, 1>>>(*_device);
        cudaDeviceSynchronize();
    } else
#endif
        printVectorBody(*this);
}

__host__ D_Array::~D_Array() {
    if (isDevice) {
        cudaFree(vals);
        cudaFree(_device);
    } else {
        delete[] vals;
    }
}
