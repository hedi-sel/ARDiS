#include "sparseDataStruct/matrix_sparse.hpp"
#include "sparseDataStruct/vector_dense.hpp"

#include <cstdio>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

void solveLinEq(cusolverSpHandle_t &handle, MatrixSparse &d_mat,
                const VectorDense &b, VectorDense &x) {
    int *singular = new int[0];
    d_mat.MakeDescriptor();
    d_mat.OperationCuSolver((void *)cusolverSpDcsrlsvqr, handle, b.vals, x.vals,
                            singular);
}