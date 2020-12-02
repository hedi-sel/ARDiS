#define GET_PROF

#include "cuda_runtime.h"
#include "include/helper/cuda/cublas_error_check.h"
#include "include/helper/cuda/cusparse_error_check.h"
#include <assert.h>
#include <stdio.h>

#include "basic_operations.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/matrix_element.hpp"
#include "helper/cuda/cuda_error_check.h"
#include "helper/cuda/cuda_reduction_operation.hpp"
#include "helper/cuda/cuda_thread_manager.hpp"

ChronoProfiler profDot;
void PrintDotProfiler() { profDot.print(); }

cusparseHandle_t cusparseHandle = NULL;
cublasHandle_t cublasHandle = NULL;

void dot(d_spmatrix &d_mat, d_vector &x, d_vector &result, bool synchronize) {
    if (!cusparseHandle)
        cusparseErrchk(cusparseCreate(&cusparseHandle));
    assert(d_mat.isDevice && x.isDevice && result.isDevice);
    if (&x == &result) {
        printf("Error: X and Result vectors should not be the same instance\n");
        return;
    }
    T one = 1.0;
    T zero = 0.0;
    size_t size = 0;
    T *buffer;
    auto mat_descr = d_mat.MakeSpDescriptor();
    auto x_descr = x.MakeDescriptor();
    auto res_descr = result.MakeDescriptor();
    cusparseErrchk(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat_descr,
        x_descr, &zero, res_descr, T_Cuda, CUSPARSE_MV_ALG_DEFAULT, &size));
    if (size > 0)
        printf("Alert! size >0 \n");
    cudaMalloc(&buffer, size);
    cusparseErrchk(cusparseSpMV(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat_descr,
        x_descr, &zero, res_descr, T_Cuda, CUSPARSE_MV_ALG_DEFAULT, buffer));
    cusparseDestroyDnVec(x_descr);
    cusparseDestroyDnVec(res_descr);
    cusparseDestroySpMat(mat_descr);
}

d_vector buffer(0);

__global__ void DotK(d_vector &x, d_vector &y, d_vector &buffer) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= x.n)
        return;
    buffer.data[i] = x.data[i] * y.data[i];
    return;
}

void dot(d_vector &x, d_vector &y, T &result, bool synchronize) {
    assert(x.isDevice && y.isDevice);
    assert(x.n == y.n);

    if (!cublasHandle)
        cublasErrchk(cublasCreate(&cublasHandle));

#ifdef USE_DOUBLE
    cublasErrchk(cublasDdot(cublasHandle, x.n, x.data, 1, y.data, 1, &result));
#else
    cublasErrchk(cublasSdot(cublasHandle, x.n, x.data, sizeof(T), y.data,
                            sizeof(T), &result));
#endif
    // dim3Pair threadblock = Make1DThreadBlock(x.n);
    // if (buffer.n < x.n)
    //     buffer.Resize(x.n);
    // else
    //     buffer.n = x.n;

    // DotK<<<threadblock.block, threadblock.thread>>>(*(d_vector *)x._device,
    //                                                 *(d_vector *)y._device,
    //                                                 *(d_vector
    //                                                 *)buffer._device);
    // ReductionOperation(buffer, sum);
    // cudaMemcpy(&result, buffer.data, sizeof(T), cudaMemcpyDeviceToDevice);
    if (synchronize) {
        gpuErrchk(cudaDeviceSynchronize());
    } else
        return;
}

__global__ void VectorSumK(d_vector &a, d_vector &b, T &alpha, d_vector &c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= a.n)
        return;
    c.data[i] = a.data[i] + b.data[i] * alpha;
};

void vector_sum(d_vector &a, d_vector &b, T &alpha, d_vector &c,
                bool synchronize) {
    assert(a.isDevice && b.isDevice);
    assert(a.n == b.n);
    dim3Pair threadblock = Make1DThreadBlock(a.n);
    VectorSumK<<<threadblock.block, threadblock.thread>>>(
        *(d_vector *)a._device, *(d_vector *)b._device, alpha,
        *(d_vector *)c._device);
    if (synchronize)
        gpuErrchk(cudaDeviceSynchronize());
}

void vector_sum(d_vector &a, d_vector &b, d_vector &c, bool synchronize) {
    hd_data<T> alpha(1.0);
    vector_sum(a, b, alpha(true), c, synchronize);
}

__device__ inline bool IsSup(MatrixElement &it_a, MatrixElement &it_b) {
    return (it_a.i == it_b.i && it_a.j > it_b.j) || it_a.i > it_b.i;
};

__device__ inline bool IsEqu(MatrixElement &it_a, MatrixElement &it_b) {
    return (it_a.i == it_b.i && it_a.j == it_b.j);
};

__device__ inline bool IsSupEqu(MatrixElement &it_a, MatrixElement &it_b) {
    return (it_a.i == it_b.i && it_a.j >= it_b.j) || it_a.i > it_b.i;
};

__global__ void SumNNZK(d_spmatrix &a, d_spmatrix &b, int *nnz) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= a.rows)
        return;
    if (i == 0)
        nnz[0] = 0;
    MatrixElement it_a(a.rowPtr[i], &a);
    MatrixElement it_b(b.rowPtr[i], &b);
    nnz[i + 1] = 0;
    while (it_a.i == i || it_b.i == i) {
        if (IsEqu(it_a, it_b)) {
            it_a.Next();
            it_b.Next();
            nnz[i + 1] += 1;
        } else if (IsSup(it_a, it_b)) {
            it_b.Next();
            nnz[i + 1] += 1;
        } else if (IsSup(it_b, it_a)) {
            it_a.Next();
            nnz[i + 1] += 1;
        } else {
            printf("Error! Nobody was iterated in SumNNZK function.\n");
            return;
        }
    }
    return;
}

__global__ void SetValuesK(d_spmatrix &a, d_spmatrix &b, T &alpha,
                           d_spmatrix &c) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= c.rows)
        return;
    MatrixElement it_a(a.rowPtr[i], &a);
    MatrixElement it_b(b.rowPtr[i], &b);
    int k = c.rowPtr[i];
    if (k >= c.nnz) {
        printf("Error! In matrix sum, at %i\n", i);
        return;
    }
    while (it_a.i == i || it_b.i == i) {
        if (IsEqu(it_a, it_b)) {
            c.colPtr[k] = it_a.j;
            c.data[k] = it_a.val[0] + alpha * it_b.val[0];
            it_a.Next();
            it_b.Next();
        } else if (IsSup(it_a, it_b)) {
            c.colPtr[k] = it_b.j;
            c.data[k] = alpha * it_b.val[0];
            it_b.Next();
        } else if (IsSup(it_b, it_a)) {
            c.colPtr[k] = it_a.j;
            c.data[k] = it_a.val[0];
            it_a.Next();
        } else {
            printf("Error! Nobody was iterated in SumNNZK function.\n");
            return;
        }
        k++;
    }
    return;
}

void matrix_sum(d_spmatrix &a, d_spmatrix &b, T &alpha, d_spmatrix &c) {
    // This method is only impleted in the specific case of CSR matrices
    assert(a.type == CSR && b.type == CSR);
    assert(a.rows == b.rows && a.cols == b.cols);
    c.rows = 1 * a.rows;
    c.cols = 1 * a.cols;
    c.type = CSR;
    int *nnzs;
    cudaMalloc(&nnzs, sizeof(int) * (a.rows + 1));
    auto tb = Make1DThreadBlock(a.rows);
    SumNNZK<<<tb.block, tb.thread>>>(*a._device, *b._device, nnzs);
    ReductionIncreasing(nnzs, a.rows + 1);
    hd_data<int> nnz(&nnzs[a.rows], true);
    c.SetNNZ(nnz());

    gpuErrchk(cudaMemcpy(c.rowPtr, nnzs, sizeof(int) * (a.rows + 1),
                         cudaMemcpyDeviceToDevice));

    SetValuesK<<<tb.block, tb.thread>>>(*a._device, *b._device, alpha,
                                        *c._device);
    gpuErrchk(cudaDeviceSynchronize());
    return;
}

void matrix_sum(d_spmatrix &a, d_spmatrix &b, d_spmatrix &c) {
    hd_data<T> d_alpha(1.0);
    matrix_sum(a, b, d_alpha(true), c);
}

__global__ void ScalarMultK(T *data, int n, T &alpha) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n)
        return;
    data[i] *= alpha;
    return;
}

void scalar_mult(d_spmatrix &a, T &alpha) {
    assert(a.isDevice);
    dim3Pair threadblock = Make1DThreadBlock(a.nnz);
    ScalarMultK<<<threadblock.block, threadblock.thread>>>(a.data, a.nnz,
                                                           alpha);
}
void scalar_mult(d_vector &a, T &alpha) {
    assert(a.isDevice);
    dim3Pair threadblock = Make1DThreadBlock(a.n);
    ScalarMultK<<<threadblock.block, threadblock.thread>>>(a.data, a.n, alpha);
}
