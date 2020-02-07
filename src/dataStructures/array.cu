#include "dataStructures/array.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/helper/vector_helper.h"
#include "hediHelper/cuda/cuda_thread_manager.hpp"
#include "helper/apply_operation.h"
#include "matrixOperations/basic_operations.hpp"

__host__ D_Array::D_Array(int n, bool isDevice) : n(n), isDevice(isDevice) {
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

__host__ void D_Array::operator=(const D_Array &other) {
    assert(isDevice == other.isDevice);
    MemFree();
    n = other.n;
    MemAlloc();
    cudaMemcpyKind memCpy =
        (other.isDevice)
            ? (isDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost
            : (isDevice) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    gpuErrchk(cudaMemcpy(vals, other.vals, sizeof(T) * n, memCpy));
}

// __host__ void D_Array::Swap(D_Array &other) {
//     assert(isDevice == other.isDevice);
//     n = other.n;
//     std::swap(_device, other._device);
//     std::swap(vals, other.vals);
//     other.MemFree();
// }

__host__ void D_Array::Resize(int n) {
    MemFree();
    this->n = n;
    MemAlloc();
}

__host__ cusparseDnVecDescr_t D_Array::MakeDescriptor() {
    cusparseDnVecDescr_t descr;
    cusparseErrchk(cusparseCreateDnVec(&descr, n, vals, T_Cuda));
    return descr;
}

__host__ T D_Array::Norm() {
    assert(isDevice);
    HDData<T> norm;
    Dot(*this, *this, norm(true));
    norm.SetHost();
    return norm();
}

__host__ void D_Array::Fill(T value) {
    auto setTo = [value] __device__(T & a) { a = value; };
    ApplyFunction(*this, setTo);
}

__host__ void D_Array::Prune(T value) {
    auto setTo = [value] __device__(T & a) {
        if (a < value)
            a = value;
    };
    ApplyFunction(*this, setTo);
}
__host__ void D_Array::PruneUnder(T value) {
    auto setTo = [value] __device__(T & a) {
        if (a > value)
            a = value;
    };
    ApplyFunction(*this, setTo);
}

__host__ __device__ void D_Array::Print(int printCount) {
    printf("[ ");
#ifndef __CUDA_ARCH__
    if (isDevice) {
        printVector<<<1, 1>>>(*_device, printCount);
        gpuErrchk(cudaDeviceSynchronize());
    } else
#endif
        printVectorBody(*this, printCount);
}

__host__ D_Array::~D_Array() { MemFree(); }

__host__ void D_Array::MemAlloc() {
    if (n > 0)
        if (isDevice) {
            gpuErrchk(cudaMalloc(&vals, n * sizeof(T)));
            gpuErrchk(cudaMalloc(&_device, sizeof(D_Array)));
            gpuErrchk(cudaMemcpy(_device, this, sizeof(D_Array),
                                 cudaMemcpyHostToDevice));
        } else {
            vals = new T[n];
        }
}

__host__ void D_Array::MemFree() {
    if (n > 0)
        if (isDevice) {
            gpuErrchk(cudaFree(vals));
            gpuErrchk(cudaFree(_device));
            gpuErrchk(cudaDeviceSynchronize());
        } else {
            delete[] vals;
        }
}