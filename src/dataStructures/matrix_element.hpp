#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <string>

#include "constants.hpp"
#include "sparse_matrix.hpp"

class MatrixElement {
  public:
    const d_spmatrix *matrix;
    int k;
    int i = 0;
    int j = 0;
    T *val;

    __host__ __device__ MatrixElement(int k, const d_spmatrix *matrix);
    __host__ __device__ MatrixElement(const d_spmatrix *matrix);
    // __host__ __device__ MatrixElement() : matrix(nullptr) {}

    __host__ __device__ bool HasNext();
    __host__ __device__ void Next();
    __host__ __device__ void Jump(int hop);

    __host__ __device__ void print() const;
    __host__ std::string ToString() const;

    __host__ __device__ void updateIandJ();
};