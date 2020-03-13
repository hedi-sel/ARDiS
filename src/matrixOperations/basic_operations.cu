#define GET_PROF

#include "cuda_runtime.h"
#include "include/hediHelper/cuda/cusparse_error_check.h"
#include <assert.h>
#include <stdio.h>

#include "basic_operations.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/matrix_element.hpp"
#include "hediHelper/cuda/cuda_error_check.h"
#include "hediHelper/cuda/cuda_reduction_operation.hpp"
#include "hediHelper/cuda/cuda_thread_manager.hpp"

ChronoProfiler profDot;
void PrintDotProfiler() { profDot.Print(); }

cusparseHandle_t handle = NULL;

void Dot(D_SparseMatrix &d_mat, D_Array &x, D_Array &result, bool synchronize) {
    profDot.Start("Prep");
    if (!handle)
        cusparseErrchk(cusparseCreate(&handle));
    assert(d_mat.isDevice && x.isDevice && result.isDevice);
    if (&x == &result) {
        printf("Error: X and Result vectors should not be the same instance\n");
        return;
    }
    profDot.Start("Alloc");
    T one = 1.0;
    T zero = 0.0;
    size_t size = 0;
    T *buffer;
    profDot.Start("BuffSize");
    auto mat_descr = d_mat.MakeSpDescriptor();
    auto x_descr = x.MakeDescriptor();
    auto res_descr = result.MakeDescriptor();
    cusparseErrchk(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat_descr, x_descr,
        &zero, res_descr, T_Cuda, CUSPARSE_MV_ALG_DEFAULT, &size));
    if (size > 0)
        printf("Alert! Size >0 \n");
    cudaMalloc(&buffer, size);
    profDot.Start("Computation");
    cusparseErrchk(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                mat_descr, x_descr, &zero, res_descr, T_Cuda,
                                CUSPARSE_MV_ALG_DEFAULT, buffer));
    cusparseDestroyDnVec(x_descr);
    cusparseDestroyDnVec(res_descr);
    cusparseDestroySpMat(mat_descr);
    profDot.End();
}

D_Array buffer(0);

__global__ void DotK(D_Array &x, D_Array &y, D_Array &buffer) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= x.n)
        return;
    buffer.vals[i] = x.vals[i] * y.vals[i];
    return;
}

void Dot(D_Array &x, D_Array &y, T &result, bool synchronize) {
    assert(x.isDevice && y.isDevice);
    assert(x.n == y.n);
    dim3Pair threadblock = Make1DThreadBlock(x.n);
    if (buffer.n < x.n)
        buffer.Resize(x.n);
    else
        buffer.n = x.n;

    DotK<<<threadblock.block, threadblock.thread>>>(*(D_Array *)x._device,
                                                    *(D_Array *)y._device,
                                                    *(D_Array *)buffer._device);
    ReductionOperation(buffer, sum);
    cudaMemcpy(&result, buffer.vals, sizeof(T), cudaMemcpyDeviceToDevice);
    if (synchronize) {
        gpuErrchk(cudaDeviceSynchronize());
    } else
        return;
}

__global__ void VectorSumK(D_Array &a, D_Array &b, T &alpha, D_Array &c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= a.n)
        return;
    c.vals[i] = a.vals[i] + b.vals[i] * alpha;
};

void VectorSum(D_Array &a, D_Array &b, T &alpha, D_Array &c, bool synchronize) {
    assert(a.isDevice && b.isDevice);
    assert(a.n == b.n);
    dim3Pair threadblock = Make1DThreadBlock(a.n);
    VectorSumK<<<threadblock.block, threadblock.thread>>>(
        *(D_Array *)a._device, *(D_Array *)b._device, alpha,
        *(D_Array *)c._device);
    if (synchronize)
        gpuErrchk(cudaDeviceSynchronize());
}

void VectorSum(D_Array &a, D_Array &b, D_Array &c, bool synchronize) {
    HDData<T> alpha(1.0);
    VectorSum(a, b, alpha(true), c, synchronize);
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

__global__ void SumNNZK(D_SparseMatrix &a, D_SparseMatrix &b, int *nnz) {
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

__global__ void SetValuesK(D_SparseMatrix &a, D_SparseMatrix &b, T &alpha,
                           D_SparseMatrix &c) {

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
            c.vals[k] = it_a.val[0] + alpha * it_b.val[0];
            it_a.Next();
            it_b.Next();
        } else if (IsSup(it_a, it_b)) {
            c.colPtr[k] = it_b.j;
            c.vals[k] = alpha * it_b.val[0];
            it_b.Next();
        } else if (IsSup(it_b, it_a)) {
            c.colPtr[k] = it_a.j;
            c.vals[k] = it_a.val[0];
            it_a.Next();
        } else {
            printf("Error! Nobody was iterated in SumNNZK function.\n");
            return;
        }
        k++;
    }
    return;
}

void MatrixSum(D_SparseMatrix &a, D_SparseMatrix &b, T &alpha,
               D_SparseMatrix &c) {
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
    HDData<int> nnz(&nnzs[a.rows], true);
    c.SetNNZ(nnz());

    gpuErrchk(cudaMemcpy(c.rowPtr, nnzs, sizeof(int) * (a.rows + 1),
                         cudaMemcpyDeviceToDevice));

    SetValuesK<<<tb.block, tb.thread>>>(*a._device, *b._device, alpha,
                                        *c._device);
    gpuErrchk(cudaDeviceSynchronize());
    return;
}

void MatrixSum(D_SparseMatrix &a, D_SparseMatrix &b, D_SparseMatrix &c) {
    HDData<T> d_alpha(1.0);
    MatrixSum(a, b, d_alpha(true), c);
}

__global__ void ScalarMultK(T *vals, int n, T &alpha) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n)
        return;
    vals[i] *= alpha;
    return;
}

void ScalarMult(D_SparseMatrix &a, T &alpha) {
    assert(a.isDevice);
    dim3Pair threadblock = Make1DThreadBlock(a.nnz);
    ScalarMultK<<<threadblock.block, threadblock.thread>>>(a.vals, a.nnz,
                                                           alpha);
}
void ScalarMult(D_Array &a, T &alpha) {
    assert(a.isDevice);
    dim3Pair threadblock = Make1DThreadBlock(a.n);
    ScalarMultK<<<threadblock.block, threadblock.thread>>>(a.vals, a.n, alpha);
}
