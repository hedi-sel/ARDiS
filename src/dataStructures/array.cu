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
    gpuErrchk(cudaMemcpy(data, m.data, sizeof(C) * n, memCpy));
}

template <typename C>
__host__ void D_Array<C>::operator=(const D_Array<C> &other) {
    Resize(other.n);
    cudaMemcpyKind memCpy =
        (other.isDevice)
            ? (isDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost
            : (isDevice) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    gpuErrchk(cudaMemcpy(data, other.data, sizeof(C) * n, memCpy));
}

template <typename C> __host__ void D_Array<C>::Resize(int n) {
    if (n == this->n)
        return;
    MemFree();
    this->n = n;
    MemAlloc();
}

template <typename C> __host__ __device__ C &D_Array<C>::At(int i) {
#ifndef __CUDA_ARCH__
    if (!isDevice)
        printf("You tried to access host data while you're on the device.\nYou "
               "should define your array as a device array.\n");
    return data[i];
#else
    if (isDevice)
        printf("You tried to access device data while you're on the host.\nYou "
               "should define your array as a host array.\n");
    return data[i];
#endif
}

template <typename C> __host__ __device__ int D_Array<C>::Size() { return n; }
template <typename C> __host__ __device__ bool D_Array<C>::IsDevice() {
    return isDevice;
}

template <typename C>
__host__ cusparseDnVecDescr_t D_Array<C>::MakeDescriptor() {
    cusparseDnVecDescr_t descr;
    cusparseErrchk(cusparseCreateDnVec(&descr, n, data, T_Cuda));
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
            gpuErrchk(cudaMalloc(&data, n * sizeof(T)));
            gpuErrchk(cudaMalloc(&_device, sizeof(D_Array<C>)));
            gpuErrchk(cudaMemcpy(_device, this, sizeof(D_Array<C>),
                                 cudaMemcpyHostToDevice));
        } else {
            data = new C[n];
        }
}

template <typename C> __host__ void D_Array<C>::MemFree() {
    if (n > 0)
        if (isDevice) {
            gpuErrchk(cudaFree(data));
            gpuErrchk(cudaFree(_device));
            gpuErrchk(cudaDeviceSynchronize());
        } else {
            delete[] data;
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
