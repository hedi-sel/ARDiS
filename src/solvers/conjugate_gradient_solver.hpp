#pragma once

#include "cuda_runtime.h"

#include "constants.hpp"
#include "cusparseOperations/dot_sparse.hpp"
#include "sparseDataStruct/double_data.hpp"
#include "sparseDataStruct/matrix_sparse.hpp"
#include "sparseDataStruct/vector_dense.hpp"

void CGSolve(D_SparseMatrix &d_mat, D_Array &b, D_Array &y, T epsilon);
void CGSolve(D_SparseMatrix &d_mat, D_Array &b, D_Array &y, T epsilon,
             D_SparseMatrix &precond);