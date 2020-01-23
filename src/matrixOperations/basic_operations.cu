#define GET_PROF

#include "cuda_runtime.h"
#include "include/hediHelper/cuda/cusparse_error_check.h"
#include <assert.h>
#include <stdio.h>

#include "dataStructures/hd_data.hpp"
#include "dataStructures/matrix_element.hpp"
#include "basic_operations.hpp"
#include "hediHelper/cuda/cuda_error_check.h"
#include "hediHelper/cuda/cuda_reduction_operation.h"
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
        printf("Error: X and Result vectors should not be the same instance");
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
        printf("Alert! Size >0)");
    cudaMalloc(&buffer, size);
    profDot.Start("Computation");
    cusparseErrchk(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                mat_descr, x_descr, &zero, res_descr, T_Cuda,
                                CUSPARSE_MV_ALG_DEFAULT, buffer));
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

    auto d_sum = [] __device__(const T &a, const T &b) { return a + b; };
    DotK<<<threadblock.block, threadblock.thread>>>(*x._device, *y._device,
                                                    *buffer._device);
    ReductionOperation<typeof(d_sum)>(buffer, d_sum);
    cudaMemcpy(&result, buffer.vals, sizeof(T), cudaMemcpyDeviceToDevice);
    if (synchronize)
        cudaDeviceSynchronize();
    else
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
        *a._device, *b._device, alpha, *c._device);
    if (synchronize)
        cudaDeviceSynchronize();
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

__global__ void SumNNZK(D_SparseMatrix &a, D_SparseMatrix &b, int &nnz) {
    MatrixElement it_a(&a);
    MatrixElement it_b(&b);
    nnz = 0;
    while (it_a.HasNext() || it_b.HasNext()) {
        if (IsEqu(it_a, it_b)) {
            it_a.Next();
            it_b.Next();
            nnz++;
        } else if (IsSup(it_a, it_b)) {
            it_b.Next();
            nnz++;
        } else if (IsSup(it_b, it_a)) {
            it_a.Next();
            nnz++;
        } else {
            printf("Error! Nobody was iterated in SumNNZK function.\n");
            return;
        }
    }
}

__global__ void AllocateSumK(D_SparseMatrix &a, D_SparseMatrix &b, T &alpha,
                             D_SparseMatrix &c) {
    // int i = threadIdx.x + blockIdx.x * blockDim.x;
    // if (i >= a.rows)
    //     return;
    MatrixElement it_a(&a);
    MatrixElement it_b(&b);
    T v = 0.0;
    c.AddElement(0, 0, v);
    MatrixElement it_c(&c);

    while (it_a.HasNext() || it_b.HasNext()) {
        if (IsSupEqu(it_a, it_b)) {
            T bVal = alpha * it_b.val[0];
            if (IsEqu(it_b, it_c))
                it_c.val[0] += bVal;
            else {
                c.AddElement(it_b.i, it_b.j, bVal);
                it_c.Next();
            }
            it_b.Next();
        }
        if (IsSupEqu(it_b, it_a)) {
            if (IsEqu(it_a, it_c))
                it_c.val[0] += it_a.val[0];
            else if (it_a.HasNext()) {
                c.AddElement(it_a.i, it_a.j, it_a.val[0]);
                it_c.Next();
            }
            it_a.Next();
        }
    }
}

void MatrixSum(D_SparseMatrix &a, D_SparseMatrix &b, T &alpha,
               D_SparseMatrix &c) {
    // This method is only impleted in the specific case of CSR matrices
    assert(a.type == CSR && b.type == CSR);
    assert(a.rows == b.rows && a.cols == b.cols);
    c.rows = 1 * a.rows;
    c.cols = 1 * a.cols;
    HDData<int> nnz;
    SumNNZK<<<1, 1>>>(*a._device, *b._device, nnz(true));
    cudaDeviceSynchronize();
    nnz.SetHost();
    c.SetNNZ(nnz());
    // dim3Pair threadblock = Make1DThreadBlock(a.rows);
    AllocateSumK<<<1, 1>>>(*a._device, *b._device, alpha, *c._device);
    cudaDeviceSynchronize();
    assert(c.IsConvertibleTo(CSR));
    c.ConvertMatrixToCSR();
    return;
}

void MatrixSum(D_SparseMatrix &a, D_SparseMatrix &b, D_SparseMatrix &c) {
    HDData<T> d_alpha(1.0);
    MatrixSum(a, b, d_alpha(true), c);
}