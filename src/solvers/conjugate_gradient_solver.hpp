#include "cuda_runtime.h"

#include "constants.hpp"
#include "cusparseOperations/dot_sparse.hpp"
#include "sparseDataStruct/double_data.hpp"
#include "sparseDataStruct/matrix_sparse.hpp"
#include "sparseDataStruct/vector_dense.hpp"

void CGSolve(MatrixSparse &d_mat, VectorDense &b, VectorDense &y, T epsilon);