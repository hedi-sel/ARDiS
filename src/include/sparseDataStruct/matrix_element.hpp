#pragma once

#include <cstdio>
#include <cuda_runtime.h>

#include "constants.hpp"
#include "matrix_sparse.hpp"

class MatrixElement {
  public:
    const MatrixSparse *matrix;
    int k;
    int i = 0;
    int j = 0;
    T *val;

    __host__ __device__ MatrixElement(int k, const MatrixSparse *matrix);
    __host__ __device__ MatrixElement(const MatrixSparse *matrix);
    // __host__ __device__ MatrixElement() : matrix(nullptr) {}

    __host__ __device__ bool HasNext();
    __host__ __device__ void Next();

    __host__ __device__ void Print() const;

    __host__ __device__ void updateIandJ();
};