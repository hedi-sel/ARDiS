#pragma once

#include <cuda_runtime.h>

#include "constants.hpp"
#include "cudaHelper/cuda_error_check.h"
#include "cuda_runtime.h"

class VectorDense {
  public:
    const int n;
    const bool isDevice;

    T *vals;

    VectorDense *_device;

    __host__ VectorDense(int, bool = false);
    __host__ VectorDense(const VectorDense &, bool copyToOtherMem = false);

    __host__ void Print();

  private:
    __host__ void MemAlloc();
};