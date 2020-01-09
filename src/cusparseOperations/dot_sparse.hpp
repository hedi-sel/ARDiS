#pragma once

#include "constants.hpp"
#include "sparseDataStruct/matrix_sparse.hpp"
#include "sparseDataStruct/vector_dense.hpp"

void Dot(MatrixSparse &d_mat, VectorDense &x, VectorDense &y,
         bool synchronize = true);
void Dot(VectorDense &x, VectorDense &y, double &result,
         bool synchronize = true);
void VectorSum(VectorDense &a, VectorDense &b, T &alpha, VectorDense &c,
               bool synchronize = true);
void VectorSum(VectorDense &a, VectorDense &b, VectorDense &c,
               bool synchronize = true);
void MatrixSum(MatrixSparse &a, MatrixSparse &b, T &alpha, MatrixSparse &c,
               bool synchronize = true);
void MatrixSum(MatrixSparse &a, MatrixSparse &b, MatrixSparse &c,
               bool synchronize = true);