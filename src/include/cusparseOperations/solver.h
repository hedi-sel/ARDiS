#include "sparseDataStruct/matrix_sparse.hpp"
#include "sparseDataStruct/vector_dense.hpp"

#include <cstdio>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

cusolverSpHandle_t cusolverSpHandle = NULL;

void solveLinEqBody(MatrixSparse &d_mat, const VectorDense &b, VectorDense &x) {
    if (cusolverSpHandle == NULL)
        cusolverErrchk(cusolverSpCreate(&cusolverSpHandle));
    int *singular = new int[0];
    d_mat.MakeDescriptor();
    d_mat.OperationCuSolver(
        (void *)cusolverSpDcsrlsvchol, // cusolverSpDcsrlsvchol
                                       // cusolverSpDcsrlsvqr
        cusolverSpHandle, b.vals, x.vals, singular);

    if (cusolverSpHandle)
        cusolverSpDestroy(cusolverSpHandle);
    cusolverSpHandle = NULL;
}