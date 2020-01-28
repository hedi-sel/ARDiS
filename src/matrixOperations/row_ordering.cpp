#include <cstdio>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include "dataStructures/sparse_matrix.hpp"
#include "hediHelper/cuda/cuda_error_check.h"
#include "hediHelper/cuda/cusparse_error_check.h"

cusparseHandle_t cusparseHandle = NULL;

void RowOrdering(D_SparseMatrix &d_mat) {
    assert(d_mat.isDevice);
    if (cusparseHandle == NULL) {
        cusparseErrchk(cusparseCreate(&cusparseHandle));
    }
    int *d_P = NULL;
    T *d_cooVals_sorted = NULL;
    size_t pBufferSizeInBytes = 0;
    void *pBuffer = NULL;

    /* step 1: ?? */
    d_mat.OperationCuSparse((void *)cusparseXcoosort_bufferSizeExt,
                            cusparseHandle, false, &pBufferSizeInBytes);

    /* step 2: allocate buffer */
    gpuErrchk(cudaMalloc(&d_P, sizeof(int) * d_mat.nnz));
    gpuErrchk(cudaMalloc(&d_cooVals_sorted, sizeof(double) * d_mat.nnz));
    gpuErrchk(cudaMalloc(&pBuffer, sizeof(char) * pBufferSizeInBytes));
    gpuErrchk(cudaDeviceSynchronize());

    /* step 3: setup permutation vector P to identity */
    cusparseErrchk(
        cusparseCreateIdentityPermutation(cusparseHandle, d_mat.nnz, d_P));

    /* step 4: sort COO format by Row */
    d_mat.OperationCuSparse((void *)cusparseXcoosortByRow, cusparseHandle,
                            false, d_P, pBuffer);

    // /* step 5: gather sorted cooVals */
#ifdef USE_DOUBLE
    cusparseErrchk(cusparseDgthr(cusparseHandle, d_mat.nnz, d_mat.vals,
                                 d_cooVals_sorted, d_P,
                                 CUSPARSE_INDEX_BASE_ZERO));
#else
    cusparseErrchk(cusparseSgthr(cusparseHandle, d_mat.nnz, d_mat.vals,
                                 d_cooVals_sorted, d_P,
                                 CUSPARSE_INDEX_BASE_ZERO));
#endif

    T *freeMe = d_mat.vals;
    d_mat.vals = d_cooVals_sorted;

    /* free resources */
    if (d_P)
        gpuErrchk(cudaFree(d_P));
    if (pBuffer)
        gpuErrchk(cudaFree(pBuffer));
    if (freeMe)
        gpuErrchk(cudaFree(freeMe));
    if (cusparseHandle)
        cusparseDestroy(cusparseHandle);
    cusparseHandle = NULL;
   gpuErrchk(cudaDeviceSynchronize());
}