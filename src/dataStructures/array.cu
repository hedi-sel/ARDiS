#include "dataStructures/array.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/helper/vector_helper.h"
#include "helper/apply_operation.h"
#include "helper/cuda/cuda_thread_manager.hpp"
#include "matrixOperations/basic_operations.hpp"
#include "sstream"

__device__ __host__ void call_error(AccessError error) {
    switch (error) {
    case AccessDeviceOnHost:
        printf("Error, trying to access device array from the host");
    case AccessHostOnDevice:
        printf("Error, trying to access host array from the device");
    }
}

template <typename C>
__host__ d_array<C>::d_array(int n, bool is_device)
    : n(n), is_device(is_device) {
    mem_alloc();
}

template <typename C>
__host__ d_array<C>::d_array(const d_array<C> &m, bool copyToOtherMem)
    : d_array<C>(m.n, m.is_device ^ copyToOtherMem) {
    cudaMemcpyKind memCpy =
        (m.is_device)
            ? (is_device) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost
            : (is_device) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    gpuErrchk(cudaMemcpy(data, m.data, sizeof(C) * n, memCpy));
}

template <typename C>
__host__ d_array<C>::d_array(d_array<C> &&other) : d_array(0, other.is_device) {
    *this = other;
}

template <typename C>
__host__ void d_array<C>::operator=(const d_array<C> &other) {
    if (is_device != other.is_device)
        if (is_device)
            throw("You cannot move an array host array into a device array");
        else
            throw("You cannot move an array device array into a host array");
    mem_free();
    n = other.n;
    n_dataholders = other.n_dataholders;
    *n_dataholders += 1;
    data = other.data;
    if (is_device)
        _device = other._device;
}

template <typename C> __host__ void d_array<C>::resize(int n) {
    mem_free();
    this->n = n;
    mem_alloc();
}

template <typename C> __host__ d_array<C>::~d_array<C>() { mem_free(); }

template <typename C> __host__ void d_array<C>::mem_alloc() {
    if (n > 0) {
        n_dataholders = new int[1];
        *n_dataholders = 1;
        if (is_device) {
            gpuErrchk(cudaMalloc(&data, n * sizeof(T)));
            gpuErrchk(cudaMalloc(&_device, sizeof(d_array<C>)));
            gpuErrchk(cudaMemcpy(_device, this, sizeof(d_array<C>),
                                 cudaMemcpyHostToDevice));
        } else {
            data = new C[n];
        }
    }
}

template <typename C> __host__ void d_array<C>::mem_free() {
    if (n > 0) {
        *n_dataholders -= 1;
        if (*n_dataholders == 0) {
            if (is_device) {
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
    if (is_device) {
        gpuErrchk(cudaDeviceSynchronize());
        print_vectorK<<<1, 1>>>(*_device, printCount);
        gpuErrchk(cudaDeviceSynchronize());
    } else
#else
    if (!is_device)
        call_error(AccessHostOnDevice);
    else
#endif
        print_vectorBody(*this, printCount);
}

template <typename C> __host__ __device__ C &d_array<C>::at(int i) {
#ifndef __CUDA_ARCH__
    if (is_device)
        call_error(AccessDeviceOnHost);
#else

    if (!is_device)
        call_error(AccessHostOnDevice);
#endif
    return data[i];
}

template <typename C> __host__ __device__ int d_array<C>::size() { return n; }

template <typename C>
__host__ cusparseDnVecDescr_t d_array<C>::make_descriptor() {
    cusparseDnVecDescr_t descr;
    cusparseErrchk(cusparseCreateDnVec(&descr, n, data, T_Cuda));
    return descr;
}

template <typename C> __host__ void d_array<C>::fill(C value) {
    auto setTo = [value] __device__(C & a) { a = value; };
    apply_func(*this, setTo);
}

#define quote(x) #x

__host__ void d_vector::prune(T value) {
    auto setTo = [value] __device__(T & a) {
        if (a < value)
            a = value;
    };
    apply_func(*this, setTo);
}
__host__ void d_vector::prune_under(T value) {
    auto setTo = [value] __device__(T & a) {
        if (a > value)
            a = value;
    };
    apply_func(*this, setTo);
}

__host__ std::string d_vector::to_string() {
    int printCount = 5;
    std::stringstream strs;
    strs << "[ ";
    T *printBuffer = new T[printCount + 1];
    cudaMemcpy(printBuffer, data, sizeof(T) * printCount,
               (is_device) ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
    cudaMemcpy(printBuffer + printCount, data + n - 1, sizeof(T),
               (is_device) ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);

    for (int i = 0; i < (n - 1) && i < printCount; i++)
        strs << printBuffer[i] << ", ";
    if (printCount < n - 1)
        strs << "... ";
    strs << printBuffer[printCount] << "]";
    delete[] printBuffer;
    return strs.str();
}
