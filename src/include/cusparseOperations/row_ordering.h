#include <cstdio>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include "cudaHelper/cuda_error_check.h"
#include "cudaHelper/cusparse_error_check.h"
#include "sparseDataStruct/matrix_sparse.hpp"

void RowOrdering(cusparseHandle_t &handle, MatrixSparse &d_mat) {
    int *d_P = NULL;
    double *d_cooVals_sorted = NULL;
    size_t pBufferSizeInBytes = 0;
    void *pBuffer = NULL;

    /* step 1: ?? */
    d_mat.OperationCuSparse((void *)cusparseXcoosort_bufferSizeExt, handle,
                            false, &pBufferSizeInBytes);

    /* step 2: allocate buffer */

    gpuErrchk(cudaMalloc(&d_P, sizeof(int) * d_mat.n_elements));
    gpuErrchk(cudaMalloc(&d_cooVals_sorted, sizeof(double) * d_mat.n_elements));
    gpuErrchk(cudaMalloc(&pBuffer, sizeof(char) * pBufferSizeInBytes));

    gpuErrchk(cudaDeviceSynchronize());

    /* step 3: setup permutation vector P to identity */
    cusparseErrchk(
        cusparseCreateIdentityPermutation(handle, d_mat.n_elements, d_P));

    /* step 4: sort COO format by Row */
    d_mat.OperationCuSparse((void *)cusparseXcoosortByRow, handle, false, d_P,
                            pBuffer);

    // /* step 5: gather sorted cooVals */
    cusparseErrchk(cusparseDgthr(handle, d_mat.n_elements, d_mat.vals,
                                 d_cooVals_sorted, d_P,
                                 CUSPARSE_INDEX_BASE_ZERO));
    cudaDeviceSynchronize();
    
    T *freeMe = d_mat.vals;
    d_mat.vals = d_cooVals_sorted;

    /* free resources */
    if (d_P)
        cudaFree(d_P);
    if (pBuffer)
        cudaFree(pBuffer);
    if (freeMe)
        cudaFree(freeMe);

    cudaDeviceSynchronize();
}