#include <cstdio>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include "cudaHelper/cusolverSP_error_check copy.h"
#include "cusparseOperations/row_ordering.h"
#include "cusparseOperations/solver.h"
#include "main.hpp"
#include "sparseDataStruct/matrix_sparse.hpp"
#include "sparseDataStruct/read_mtx_file.h"
#include "sparseDataStruct/vector_dense.hpp"

__global__ void testkernel() { printf("%i \n", 1); }

py::array_t<double> launch() {
    VectorDense a(10, false);
    for (int i = 0; i < 10; i++)
        a.vals[i] = i + 10;

    MatrixSparse matrix = ReadFromFile("matrix/test.mtx");

    MatrixSparse d_mat(matrix, true);

    cusparseHandle_t cusparseHandle = NULL;
    cusparseErrchk(cusparseCreate(&cusparseHandle));

    cusolverSpHandle_t cusolverHandle = NULL;
    cusolverErrchk(cusolverSpCreate(&cusolverHandle));

    // RowOrdering(cusparseHandle, d_mat);
    // d_mat.ToCompressedDataType(CSR);

    VectorDense b(d_mat.i_size, true);
    T *ar = new T[b.n];
    for (int i = 0; i < b.n; i++)
        ar[i] = 1.0;
    gpuErrchk(cudaMemcpy(b.vals, ar, sizeof(T) * b.n, cudaMemcpyHostToDevice));
    b.Print();
    d_mat.Print();

    solveLinEq(cusolverHandle, d_mat, b);

    // startTestSolve();

    MatrixSparse result_mat(d_mat, true);
    // result_mat.Get(3).Print();

    if (cusparseHandle)
        cusparseDestroy(cusparseHandle);
    if (cusolverHandle)
        cusolverSpDestroy(cusolverHandle);
    return py::array_t({2, 5}, a.vals);
}
