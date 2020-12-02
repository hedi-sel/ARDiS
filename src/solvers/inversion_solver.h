#include "dataStructures/array.hpp"
#include "dataStructures/sparse_matrix.hpp"

#include <cstdio>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

cusolverSpHandle_t cusolverSpHandle = NULL;

void solveLinEqBody(d_spmatrix &d_mat, const d_vector &b, d_vector &x) {
    if (cusolverSpHandle == NULL)
        cusolverErrchk(cusolverSpCreate(&cusolverSpHandle));
    int *singular = new int[0];
    auto descr = d_mat.MakeDescriptor();
    d_mat.OperationCuSolver(
        (void *)cusolverSpDcsrlsvchol, // cusolverSpDcsrlsvchol
                                       // cusolverSpDcsrlsvqr
        cusolverSpHandle, descr, b.data, x.data, singular);
    cudaFree(descr);
    assert(singular[0] == -1);

    delete[] singular;

    if (cusolverSpHandle)
        cusolverSpDestroy(cusolverSpHandle);
    cusolverSpHandle = NULL;
}