#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <nvfunctional>

#include "constants.hpp"
#include "hediHelper/cuda/cuda_error_check.h"
#include "hediHelper/cuda/cuda_thread_manager.hpp"
#include "hediHelper/cuda/cusparse_error_check.h"

template <typename C> class D_Array {
  public:
    int n;
    const bool isDevice;

    C *vals;

    D_Array *_device;

    __host__ D_Array(int, bool = true);
    __host__ D_Array(const D_Array &, bool copyToOtherMem = false);
    __host__ void operator=(const D_Array &);
    __host__ void Swap(D_Array &);

    __host__ void Resize(int);

    __host__ cusparseDnVecDescr_t MakeDescriptor();

    __host__ void Fill(C value);
    __host__ ~D_Array();

  private:
    __host__ void MemAlloc();
    __host__ void MemFree();
};

// typedef D_Array<T> D_Vector;

template class D_Array<T>;

class D_Vector : public D_Array<T> {
  public:
    using D_Array<T>::D_Array;
    __host__ __device__ void Print(int printCount = 5);
    __host__ void Prune(T value = 0);
    __host__ void PruneUnder(T value = 0);
};

template class D_Array<int>;
template class D_Array<D_Vector *>;
