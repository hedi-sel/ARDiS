#include "sparseDataStruct/helper/vector_helper.h"
#include "sparseDataStruct/vector_dense.hpp"

__host__ VectorDense::VectorDense(int n, bool isDevice)
    : n(n), isDevice(isDevice) {
    MemAlloc();
}

__host__ VectorDense::VectorDense(const VectorDense &m, bool copyToOtherMem)
    : VectorDense(m.n, m.isDevice ^ copyToOtherMem) {
    cudaMemcpyKind memCpy =
        (m.isDevice)
            ? (isDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost
            : (isDevice) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    gpuErrchk(cudaMemcpy(vals, m.vals, sizeof(T) * n, memCpy));
}

__host__ void VectorDense::MemAlloc() {
    if (isDevice) {
        gpuErrchk(cudaMalloc(&vals, n * sizeof(T)));
        gpuErrchk(cudaMalloc(&_device, sizeof(VectorDense)));
        gpuErrchk(cudaMemcpy(_device, this, sizeof(VectorDense),
                             cudaMemcpyHostToDevice));
    } else {
        vals = new T[n];
    }
}

__host__ __device__ void VectorDense::Print() {
    printf("[ ");
#ifndef __CUDA_ARCH__
    if (isDevice) {
        printVector<<<1, 1>>>(_device);
        cudaDeviceSynchronize();
    } else
#endif
        printVectorBody(this);
}

__host__ VectorDense::~VectorDense() {
    if (isDevice) {
        cudaFree(vals);
        cudaFree(_device);
    } else {
        delete[] vals;
    }
}
