#include "dataStructures/array.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/helper/vector_helper.h"
#include "hediHelper/cuda/cuda_thread_manager.hpp"
#include "helper/apply_operation.h"
#include "matrixOperations/basic_operations.hpp"

template <typename C>
__host__ D_Vector<C>::D_Vector(int n, bool isDevice)
    : n(n), isDevice(isDevice) {
    MemAlloc();
}

template <typename C>
__host__ D_Vector<C>::D_Vector(const D_Vector<C> &m, bool copyToOtherMem)
    : D_Vector<C>(m.n, m.isDevice ^ copyToOtherMem) {
    cudaMemcpyKind memCpy =
        (m.isDevice)
            ? (isDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost
            : (isDevice) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    gpuErrchk(cudaMemcpy(vals, m.vals, sizeof(C) * n, memCpy));
}

template <typename C>
__host__ void D_Vector<C>::operator=(const D_Vector<C> &other) {
    Resize(other.n);
    cudaMemcpyKind memCpy =
        (other.isDevice)
            ? (isDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost
            : (isDevice) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    gpuErrchk(cudaMemcpy(vals, other.vals, sizeof(C) * n, memCpy));
}

// template <typename C>  __host__ void D_Vector<C>::Swap(D_Vector<C> &other) {
//     assert(isDevice == other.isDevice);
//     n = other.n;
//     std::swap(_device, other._device);
//     std::swap(vals, other.vals);
//     other.MemFree();
// }

template <typename C> __host__ void D_Vector<C>::Resize(int n) {
    if (n == this->n)
        return;
    MemFree();
    this->n = n;
    MemAlloc();
}

template <typename C>
__host__ cusparseDnVecDescr_t D_Vector<C>::MakeDescriptor() {
    cusparseDnVecDescr_t descr;
    cusparseErrchk(cusparseCreateDnVec(&descr, n, vals, T_Cuda));
    return descr;
}

template <typename C> __host__ void D_Vector<C>::Fill(C value) {
    auto setTo = [value] __device__(C & a) { a = value; };
    ApplyFunction(*this, setTo);
}

template <typename C> __host__ D_Vector<C>::~D_Vector<C>() { MemFree(); }

#define quote(x) #x

template <typename C> __host__ void D_Vector<C>::MemAlloc() {
    if (n > 0)
        if (isDevice) {
            gpuErrchk(cudaMalloc(&vals, n * sizeof(T)));
            gpuErrchk(cudaMalloc(&_device, sizeof(D_Vector<C>)));
            gpuErrchk(cudaMemcpy(_device, this, sizeof(D_Vector<C>),
                                 cudaMemcpyHostToDevice));
        } else {
            vals = new C[n];
        }
}

template <typename C> __host__ void D_Vector<C>::MemFree() {
    if (n > 0)
        if (isDevice) {
            gpuErrchk(cudaFree(vals));
            gpuErrchk(cudaFree(_device));
            gpuErrchk(cudaDeviceSynchronize());
        } else {
            delete[] vals;
        }
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
#ifndef __CUDA_ARCH__
    if (isDevice) {
        printVector<<<1, 1>>>(*_device, printCount);
        gpuErrchk(cudaDeviceSynchronize());
    } else
#endif
        printVectorBody<T>(*this, printCount);
}
