#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <nvfunctional>

#include "constants.hpp"
#include "helper/cuda/cuda_error_check.h"
#include "helper/cuda/cuda_thread_manager.hpp"
#include "helper/cuda/cusparse_error_check.h"

template <typename C> class D_Array {
  public:
    int n;
    const bool isDevice;

    C *data;

    D_Array *_device;

    // Constructors
    __host__ D_Array(int = 0, bool = true);
    __host__ D_Array(const D_Array &, bool copyToOtherMem = false);

    // Manipulation
    __host__ void operator=(const D_Array &);
    __host__ void Resize(int);
    __host__ void Fill(C value);

    // Accessors
    __host__ __device__ C &at(int i);
    __host__ __device__ int size();
    __host__ __device__ bool IsDevice();
    __host__ cusparseDnVecDescr_t MakeDescriptor();

    __host__ ~D_Array();

  private:
    __host__ void MemAlloc();
    __host__ void MemFree();
};

// typedef D_Array<T> D_Vector;

template class D_Array<T>;
template class D_Array<bool>;
template class D_Array<int>;

class D_Vector : public D_Array<T> {
  public:
    using D_Array<T>::D_Array;
    __host__ __device__ void Print(int printCount = 5);
    __host__ std::string ToString();
    __host__ void Prune(T value = 0);
    __host__ void PruneUnder(T value = 0);
};

template class D_Array<D_Vector *>;
