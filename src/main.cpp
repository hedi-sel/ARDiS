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

MatrixSparse *matrix;
MatrixSparse *d_mat;

cusparseHandle_t cusparseHandle = NULL;
cusolverSpHandle_t cusolverHandle = NULL;

void Initialization() {
    cusparseErrchk(cusparseCreate(&cusparseHandle));
    cusolverErrchk(cusolverSpCreate(&cusolverHandle));
}

void LoadMatrixFromFile(char *path, bool sendToGPU) {
    Initialization();
    matrix = ReadFromFile(path);
    if (sendToGPU)
        SendMatrixToGpuMemory();
}

py::array_t<double> SolveLinEq(py::array_t<double> bVec) {
    assert(bVec.size() == d_mat->i_size);
    VectorDense b(d_mat->i_size, true);
    gpuErrchk(cudaMemcpy(b.vals, bVec.data(), sizeof(T) * b.n,
                         cudaMemcpyHostToDevice));

    VectorDense x(d_mat->i_size, true);

    solveLinEq(cusolverHandle, *d_mat, b, x);
    VectorDense result(x, true);

    if (cusparseHandle)
        cusparseDestroy(cusparseHandle);
    if (cusolverHandle)
        cusolverSpDestroy(cusolverHandle);
    return py::array_t(result.n, result.vals);
}

void SendMatrixToGpuMemory() { d_mat = new MatrixSparse(*matrix, true); }

void ConvertMatrixToCSR() {
    RowOrdering(cusparseHandle, *d_mat);
    d_mat->ToCompressedDataType(CSR);
}

void PrintMatrix(bool printGpuVersion) {
    if (printGpuVersion)
        d_mat->Print();
    else
        matrix->Print();
}
