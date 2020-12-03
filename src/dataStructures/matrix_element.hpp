#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <string>

#include "constants.hpp"
#include "sparse_matrix.hpp"

class matrix_elm {
  public:
    const d_spmatrix *matrix;
    int k;
    int i = 0;
    int j = 0;
    T *val;

    __host__ __device__ matrix_elm(int k, const d_spmatrix *matrix);
    __host__ __device__ matrix_elm(const d_spmatrix *matrix);
    // __host__ __device__ matrix_elm() : matrix(nullptr) {}

    __host__ __device__ bool has_next();
    __host__ __device__ void next();
    __host__ __device__ void jump(int hop);

    __host__ __device__ void print() const;
    __host__ std::string to_string() const;

    __host__ __device__ void updateIandJ();
};