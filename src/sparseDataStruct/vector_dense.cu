#include "sparseDataStruct/helper/vector_helper.h"
#include "sparseDataStruct/vector_dense.hpp"

__host__ VectorDense::VectorDense(int n, bool isDevice)
    : n(n), isDevice(isDevice) {
    MemAlloc();
}

__host__ VectorDense::VectorDense(const VectorDense &m, bool copyToOtherMem)
    : VectorDense(n, m.isDevice ^ copyToOtherMem) {
    cudaMemcpyKind memCpy =
        (m.isDevice)
            ? (isDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost
            : (isDevice) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    gpuErrchk(cudaMemcpy(vals, m.vals, sizeof(T) * n, memCpy));

    gpuErrchk(cudaMalloc(&_device, sizeof(VectorDense)));
    gpuErrchk(
        cudaMemcpy(_device, this, sizeof(VectorDense), cudaMemcpyHostToDevice));
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

__host__ void VectorDense::Print() {
    printf("Vector values: ");
    if (isDevice) {
        printVector<<<1, 1>>>(_device);
        cudaDeviceSynchronize();
    } else {
        printVectorBody(this);
    }
}
