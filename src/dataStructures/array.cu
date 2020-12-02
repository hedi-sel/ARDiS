#include "dataStructures/array.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/helper/vector_helper.h"
#include "helper/apply_operation.h"
#include "helper/cuda/cuda_thread_manager.hpp"
#include "matrixOperations/basic_operations.hpp"
#include "sstream"

__device__ __host__ void CallError(AccessError error) {
    switch (error) {
    case AccessDeviceOnHost:
        printf("Error, trying to access device array from the host");
    case AccessHostOnDevice:
        printf("Error, trying to access host array from the device");
    }
}

template <typename C>
__host__ d_array<C>::d_array(int n, bool isDevice) : n(n), isDevice(isDevice) {
    MemAlloc();
}

template <typename C>
__host__ d_array<C>::d_array(const d_array<C> &m, bool copyToOtherMem)
    : d_array<C>(m.n, m.isDevice ^ copyToOtherMem) {
    cudaMemcpyKind memCpy =
        (m.isDevice)
            ? (isDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost
            : (isDevice) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    gpuErrchk(cudaMemcpy(data, m.data, sizeof(C) * n, memCpy));
}

template <typename C>
__host__ d_array<C>::d_array(d_array<C> &&other) : d_array(0, other.isDevice) {
    *this = other;
}

template <typename C>
__host__ void d_array<C>::operator=(const d_array<C> &other) {
    if (isDevice != other.isDevice)
        if (isDevice)
            throw("You cannot move an array host array into a device array");
        else
            throw("You cannot move an array device array into a host array");
    MemFree();
    n = other.n;
    n_dataholders = other.n_dataholders;
    *n_dataholders += 1;
    data = other.data;
    if (isDevice)
        _device = other._device;
}

template <typename C> __host__ void d_array<C>::Resize(int n) {
    MemFree();
    this->n = n;
    MemAlloc();
}

template <typename C> __host__ d_array<C>::~d_array<C>() { MemFree(); }

template <typename C> __host__ void d_array<C>::MemAlloc() {
    if (n > 0) {
        n_dataholders = new int[1];
        *n_dataholders = 1;
        if (isDevice) {
            gpuErrchk(cudaMalloc(&data, n * sizeof(T)));
            gpuErrchk(cudaMalloc(&_device, sizeof(d_array<C>)));
            gpuErrchk(cudaMemcpy(_device, this, sizeof(d_array<C>),
                                 cudaMemcpyHostToDevice));
        } else {
            data = new C[n];
        }
    }
}

template <typename C> __host__ void d_array<C>::MemFree() {
    if (n > 0) {
        *n_dataholders -= 1;
        if (*n_dataholders == 0) {
            if (isDevice) {
                gpuErrchk(cudaFree(data));
                gpuErrchk(cudaFree(_device));
                gpuErrchk(cudaDeviceSynchronize());
            } else {
                delete[] data;
            }
        }
    }
}

template <typename C>
__host__ __device__ void d_array<C>::print(int printCount) const {
#ifndef __CUDA_ARCH__
    if (isDevice) {
        gpuErrchk(cudaDeviceSynchronize());
        printVectorK<<<1, 1>>>(*_device, printCount);
        gpuErrchk(cudaDeviceSynchronize());
    } else
#else
    if (!isDevice)
        CallError(AccessHostOnDevice);
    else
#endif
        printVectorBody(*this, printCount);
}

template <typename C> __host__ __device__ C &d_array<C>::at(int i) {
#ifndef __CUDA_ARCH__
    if (isDevice)
        CallError(AccessDeviceOnHost);
#else

    if (!isDevice)
        CallError(AccessHostOnDevice);
#endif
    return data[i];
}

template <typename C> __host__ __device__ int d_array<C>::size() { return n; }
template <typename C> __host__ __device__ bool d_array<C>::IsDevice() {
    return isDevice;
}

template <typename C>
__host__ cusparseDnVecDescr_t d_array<C>::MakeDescriptor() {
    cusparseDnVecDescr_t descr;
    cusparseErrchk(cusparseCreateDnVec(&descr, n, data, T_Cuda));
    return descr;
}

template <typename C> __host__ void d_array<C>::Fill(C value) {
    auto setTo = [value] __device__(C & a) { a = value; };
    ApplyFunction(*this, setTo);
}

#define quote(x) #x

__host__ void d_vector::prune(T value) {
    auto setTo = [value] __device__(T & a) {
        if (a < value)
            a = value;
    };
    ApplyFunction(*this, setTo);
}
__host__ void d_vector::prune_under(T value) {
    auto setTo = [value] __device__(T & a) {
        if (a > value)
            a = value;
    };
    ApplyFunction(*this, setTo);
}

__host__ std::string d_vector::ToString() {
    int printCount = 5;
    std::stringstream strs;
    strs << "[ ";
    T *printBuffer = new T[printCount + 1];
    cudaMemcpy(printBuffer, data, sizeof(T) * printCount,
               (isDevice) ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
    cudaMemcpy(printBuffer + printCount, data + n - 1, sizeof(T),
               (isDevice) ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);

    for (int i = 0; i < (n - 1) && i < printCount; i++)
        strs << printBuffer[i] << ", ";
    if (printCount < n - 1)
        strs << "... ";
    strs << printBuffer[printCount] << "]";
    delete[] printBuffer;
    return strs.str();
}
