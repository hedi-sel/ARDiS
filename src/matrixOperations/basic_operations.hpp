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
void Dot(D_SparseMatrix &d_mat, D_Vector &x, D_Vector &y,
         bool synchronize = true);
void Dot(D_Vector &x, D_Vector &y, T &result, bool synchronize = true);

// Computes C = A + alpha*B
void VectorSum(D_Vector &a, D_Vector &b, T &alpha, D_Vector &c,
               bool synchronize = true);
// Computes C = A + B
void VectorSum(D_Vector &a, D_Vector &b, D_Vector &c, bool synchronize = true);

void MatrixSum(D_SparseMatrix &a, D_SparseMatrix &b, T &alpha,
               D_SparseMatrix &c);
void MatrixSum(D_SparseMatrix &a, D_SparseMatrix &b, D_SparseMatrix &c);

void ScalarMult(D_SparseMatrix &a, T &alpha);
void ScalarMult(D_Vector &a, T &alpha);

void PrintDotProfiler();
