#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <nvfunctional>

#include "constants.hpp"
#include "hediHelper/cuda/cuda_error_check.h"
#include "hediHelper/cuda/cuda_thread_manager.hpp"
#include "hediHelper/cuda/cusparse_error_check.h"

template <typename C> class D_Vector {
  public:
    int n;
    const bool isDevice;

    C *vals;

    D_Vector *_device;

    __host__ D_Vector(int, bool = true);
    __host__ D_Vector(const D_Vector &, bool copyToOtherMem = false);
    __host__ void operator=(const D_Vector &);
    __host__ void Swap(D_Vector &);

    __host__ void Resize(int);

    __host__ cusparseDnVecDescr_t MakeDescriptor();

    __host__ void Fill(C value);
    __host__ ~D_Vector();

  private:
    __host__ void MemAlloc();
    __host__ void MemFree();
};

// typedef D_Vector<T> D_Array;

template class D_Vector<T>;

class D_Array : public D_Vector<T> {
  public:
    using D_Vector<T>::D_Vector;
    __host__ __device__ void Print(int printCount = 5);
    __host__ void Prune(T value = 0);
    __host__ void PruneUnder(T value = 0);
};

template class D_Vector<int>;
template class D_Vector<D_Array *>;
