#include "sparseDataStruct/matrix_sparse.hpp"
#include "sparseDataStruct/vector_dense.hpp"

#include <cstdio>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

void solveLinEq(cusolverSpHandle_t &handle, MatrixSparse &d_mat,
                VectorDense &b) {

    VectorDense x(d_mat.i_size, true);
    int *singular = new int[0];

    d_mat.MakeDescriptor();

    // cusolverSpDcsrlsvchol(handle, d_mat.i_size, d_mat.n_elements,
    // d_mat.descr, d_mat.vals, d_mat.rowPtr, d_mat.colPtr, b.vals,
    //                       0.0, 0, x, singular);
    d_mat.OperationCuSolver((void *)cusolverSpDcsrlsvchol, handle, b.vals,
                            x.vals, singular);

    x.Print();
    printf("%i \n", *singular);
}