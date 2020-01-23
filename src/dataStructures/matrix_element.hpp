#pragma once

#include <cstdio>
#include <cuda_runtime.h>

#include "constants.hpp"
#include "sparse_matrix.hpp"

class MatrixElement {
  public:
    const D_SparseMatrix *matrix;
    int k;
    int i = 0;
    int j = 0;
    T *val;

    __host__ __device__ MatrixElement(int k, const D_SparseMatrix *matrix);
    __host__ __device__ MatrixElement(const D_SparseMatrix *matrix);
    // __host__ __device__ MatrixElement() : matrix(nullptr) {}

    __host__ __device__ bool HasNext();
    __host__ __device__ void Next();
    __host__ __device__ void Jump(int hop);

    __host__ __device__ void Print() const;

    __host__ __device__ void updateIandJ();
};