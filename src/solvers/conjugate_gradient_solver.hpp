#pragma once

#include "cuda_runtime.h"

#include "constants.hpp"
#include "matrixOperations/basic_operations.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "dataStructures/array.hpp"

void CGSolve(D_SparseMatrix &d_mat, D_Array &b, D_Array &y, T epsilon);
void CGSolve(D_SparseMatrix &d_mat, D_Array &b, D_Array &y, T epsilon,
             D_SparseMatrix &precond);