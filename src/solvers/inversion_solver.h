#include "dataStructures/sparse_matrix.hpp"
#include "dataStructures/array.hpp"

#include <cstdio>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

cusolverSpHandle_t cusolverSpHandle = NULL;

void solveLinEqBody(D_SparseMatrix &d_mat, const D_Array &b, D_Array &x) {
    if (cusolverSpHandle == NULL)
        cusolverErrchk(cusolverSpCreate(&cusolverSpHandle));
    int *singular = new int[0];
    d_mat.OperationCuSolver(
        (void *)cusolverSpDcsrlsvchol, // cusolverSpDcsrlsvchol
                                       // cusolverSpDcsrlsvqr
        cusolverSpHandle, d_mat.MakeDescriptor(), b.vals, x.vals, singular);
    assert(singular[0] == -1);

    delete[] singular;

    if (cusolverSpHandle)
        cusolverSpDestroy(cusolverSpHandle);
    cusolverSpHandle = NULL;
}