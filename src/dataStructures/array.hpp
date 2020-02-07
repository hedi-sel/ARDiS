#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <nvfunctional>

#include "constants.hpp"
#include "hediHelper/cuda/cuda_error_check.h"
#include "hediHelper/cuda/cuda_thread_manager.hpp"
#include "hediHelper/cuda/cusparse_error_check.h"

class D_Array {
  public:
    int n;
    const bool isDevice;

    T *vals;

    D_Array *_device;

    __host__ D_Array(int, bool = true);
    __host__ D_Array(const D_Array &, bool copyToOtherMem = false);
    __host__ void operator=(const D_Array &);
    __host__ void Swap(D_Array &);

    __host__ void Resize(int);

    __host__ cusparseDnVecDescr_t MakeDescriptor();

    __host__ T Norm();
    __host__ void Fill(T value);
    __host__ void Prune(T value = 0);
    __host__ void PruneUnder(T value = 0);

    __host__ __device__ void Print(int printCount = 5);

    __host__ ~D_Array();

  private:
    __host__ void MemAlloc();
    __host__ void MemFree();
};