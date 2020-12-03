#pragma once

#include <cusparse.h>

#include "constants.hpp"
#include "dataStructures/array.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "helper/chrono_profiler.hpp"

// cudaDataType T_Cuda;
#ifdef USE_DOUBLE
const cudaDataType T_Cuda = CUDA_R_64F;
#else
const cudaDataType T_Cuda = CUDA_R_32F;
#endif
void dot(d_spmatrix &d_mat, d_vector &x, d_vector &y, bool synchronize = true);
void dot(d_vector &x, d_vector &y, T &result, bool synchronize = true);

// Computes C = A + alpha*B
void vector_sum(d_vector &a, d_vector &b, T &alpha, d_vector &c,
                bool synchronize = true);
// Computes C = A + B
void vector_sum(d_vector &a, d_vector &b, d_vector &c, bool synchronize = true);

void matrix_sum(d_spmatrix &a, d_spmatrix &b, T &alpha, d_spmatrix &c);
void matrix_sum(d_spmatrix &a, d_spmatrix &b, d_spmatrix &c);

void scalar_mult(d_spmatrix &a, T &alpha);
void scalar_mult(d_vector &a, T &alpha);

void print_dotprofiler();
