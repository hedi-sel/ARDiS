#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <cstdio>

#include "main.hpp"
#include "cudaHelper/cuda_error_check.h"
#include "cudaHelper/cusparse_error_check.h"
#include "sparseDataStruct/matrix_sparse.hpp"

__global__ void testkernel()
{
    printf("%i \n", 1);
}

py::array_t<double> launch()
{
    dim3 t(0, 1, 2);

    double *a = new double[10];
    for (int i = 0; i < 10; i++)
        a[i] = i + 10;

    MatrixSparse matrix(3, 3, 4, COO);
    matrix.AddElement(0, 2, 1, 4.0);
    matrix.AddElement(1, 1, 1, 5.0);
    matrix.AddElement(2, 0, 0, 1.0);
    matrix.AddElement(3, 0, 1, 2.0);

    MatrixSparse d_mat(matrix, true);
    printf("%i %i %i\n", d_mat.i_size, d_mat.j_size, d_mat.n_elements);

    cusparseHandle_t handle = NULL;
    cudaStream_t stream = NULL;

    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
    /*
     * A is a 3x3 sparse matrix
     *     | 1 2 0 |
     * A = | 0 5 0 |
     *     | 0 8 0 |
     */
    int *d_P = NULL;
    double *d_cooVals_sorted = NULL;
    size_t pBufferSizeInBytes = 0;
    void *pBuffer = NULL;

    /* step 1: create cusparse handle, bind a stream */
    // gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // cusparseErrchk(cusparseCreate(&handle));

    // d_mat.OperationCuSparse((void *)cusparseXcoosort_bufferSizeExt, handle, &pBufferSizeInBytes);

    //     /* step 2: allocate buffer */

    //     printf("pBufferSizeInBytes = %lld bytes \n", (long long)pBufferSizeInBytes);

    //     cudaMalloc(&d_P, sizeof(int) * d_mat.n_elements);
    //     cudaMalloc(&d_cooVals_sorted, sizeof(double) * d_mat.n_elements);
    //     cudaMalloc(&pBuffer, sizeof(char) * pBufferSizeInBytes);

    //     cudaStat4 = cudaDeviceSynchronize();

    //     /* step 3: setup permutation vector P to identity */
    //     status = cusparseCreateIdentityPermutation(handle, d_mat.n_elements, d_P);
    //     assert(CUSPARSE_STATUS_SUCCESS == status);

    //     /* step 4: sort COO format by Row */
    //     d_mat.OperationCuSparse((void *)cusparseXcoosortByRow, handle, pBuffer);

    //     /* step 5: gather sorted cooVals */
    //     status = cusparseDgthr(
    //         handle,
    //         d_mat.n_elements,
    //         d_mat.vals,
    //         d_cooVals_sorted,
    //         d_P,
    //         CUSPARSE_INDEX_BASE_ZERO);
    //     assert(CUSPARSE_STATUS_SUCCESS == status);

        cudaDeviceSynchronize(); /* wait until the computation is done */
    MatrixSparse result_mat(d_mat, true);
    cudaDeviceSynchronize();

    printf("sorted coo: \n");
    for (int j = 0; j < result_mat.n_elements; j++)
    {
        result_mat.Get(j).Print();
    }

    //     /* free resources */
    //     if (d_P)
    //         cudaFree(d_P);
    //     if (d_cooVals_sorted)
    //         cudaFree(d_cooVals_sorted);
    //     if (pBuffer)
    //         cudaFree(pBuffer);
    //     if (handle)
    //         cusparseDestroy(handle);
    //     if (stream)
    //         cudaStreamDestroy(stream);
    //     cudaDeviceReset();
    return py::array_t({2, 5}, a);
}
