#pragma once

#include <cuda_runtime.h>

#include "constants.hpp"
#include "cudaHelper/cuda_error_check.h"
#include "cuda_runtime.h"
// #include "include/pybind11_include.hpp"

class D_Array {
  public:
    const int n;
    const bool isDevice;

    T *vals;

    D_Array *_device;

    __host__ D_Array(int, bool = true);
    __host__ D_Array(const D_Array &, bool copyToOtherMem = false);

    __host__ __device__ void Print();

    __host__ ~D_Array();

  private:
    __host__ void MemAlloc();
};