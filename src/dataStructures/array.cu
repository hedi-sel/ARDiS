#include "dataStructures/array.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/helper/vector_helper.h"
#include "hediHelper/cuda/cuda_thread_manager.hpp"
#include "helper/apply_operation.h"
#include "matrixOperations/basic_operations.hpp"

template <typename C>
__host__ D_Array<C>::D_Array(int n, bool isDevice) : n(n), isDevice(isDevice) {
    MemAlloc();
}

template <typename C>
__host__ D_Array<C>::D_Array(const D_Array<C> &m, bool copyToOtherMem)
    : D_Array<C>(m.n, m.isDevice ^ copyToOtherMem) {
    cudaMemcpyKind memCpy =
        (m.isDevice)
            ? (isDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost
            : (isDevice) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    gpuErrchk(cudaMemcpy(vals, m.vals, sizeof(C) * n, memCpy));
}

template <typename C>
__host__ void D_Array<C>::operator=(const D_Array<C> &other) {
    Resize(other.n);
    cudaMemcpyKind memCpy =
        (other.isDevice)
            ? (isDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost
            : (isDevice) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    gpuErrchk(cudaMemcpy(vals, other.vals, sizeof(C) * n, memCpy));
}

// template <typename C>  __host__ void D_Array<C>::Swap(D_Array<C> &other)
// {
//     assert(isDevice == other.isDevice);
//     n = other.n;
//     std::swap(_device, other._device);
//     std::swap(vals, other.vals);
//     other.MemFree();
// }

template <typename C> __host__ void D_Array<C>::Resize(int n) {
    if (n == this->n)
        return;
    MemFree();
    this->n = n;
    MemAlloc();
}

template <typename C>
__host__ cusparseDnVecDescr_t D_Array<C>::MakeDescriptor() {
    cusparseDnVecDescr_t descr;
    cusparseErrchk(cusparseCreateDnVec(&descr, n, vals, T_Cuda));
    return descr;
}

template <typename C> __host__ void D_Array<C>::Fill(C value) {
    auto setTo = [value] __device__(C & a) { a = value; };
    ApplyFunction(*this, setTo);
}

template <typename C> __host__ D_Array<C>::~D_Array<C>() { MemFree(); }

#define quote(x) #x

template <typename C> __host__ void D_Array<C>::MemAlloc() {
    if (n > 0)
        if (isDevice) {
            gpuErrchk(cudaMalloc(&vals, n * sizeof(T)));
            gpuErrchk(cudaMalloc(&_device, sizeof(D_Array<C>)));
            gpuErrchk(cudaMemcpy(_device, this, sizeof(D_Array<C>),
                                 cudaMemcpyHostToDevice));
        } else {
            vals = new C[n];
        }
}

template <typename C> __host__ void D_Array<C>::MemFree() {
    if (n > 0)
        if (isDevice) {
            gpuErrchk(cudaFree(vals));
            gpuErrchk(cudaFree(_device));
            gpuErrchk(cudaDeviceSynchronize());
        } else {
            delete[] vals;
        }
}

__host__ void D_Vector::Prune(T value) {
    auto setTo = [value] __device__(T & a) {
        if (a < value)
            a = value;
    };
    ApplyFunction(*this, setTo);
}
__host__ void D_Vector::PruneUnder(T value) {
    auto setTo = [value] __device__(T & a) {
        if (a > value)
            a = value;
    };
    ApplyFunction(*this, setTo);
}

__host__ __device__ void D_Vector::Print(int printCount) {
#ifndef __CUDA_ARCH__
    if (isDevice) {
        printVector<<<1, 1>>>(*_device, printCount);
        gpuErrchk(cudaDeviceSynchronize());
    } else
#endif
        printVectorBody<T>(*this, printCount);
}
