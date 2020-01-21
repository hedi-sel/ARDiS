#include "cuda_runtime.h"
#include <assert.h>
#include <stdio.h>

#include "cudaHelper/cuda_error_check.h"
#include "dot_sparse.hpp"
#include "sparseDataStruct/double_data.hpp"
#include "sparseDataStruct/matrix_element.hpp"

#include "cudaHelper/cuda_thread_manager.h"

__global__ void DotK(D_SparseMatrix &d_mat, D_Array &x, D_Array &y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= d_mat.i_size)
        return;
    MatrixElement it(d_mat.rowPtr[i], &d_mat);
    y.vals[i] = 0;
    do {
        y.vals[i] += it.val[0] * x.vals[it.j];
        it.Next();
    } while (it.i == i && it.HasNext());
}

void Dot(D_SparseMatrix &d_mat, D_Array &x, D_Array &result, bool synchronize) {
    assert(d_mat.isDevice && x.isDevice && result.isDevice);
    if (&x == &result) {
        printf("Error: X and Result vectors should not be the same instance");
        return;
    }
    dim3Pair threadblock = Make1DThreadBlock(d_mat.i_size);
    DotK<<<threadblock.block, threadblock.thread>>>(*d_mat._device, *x._device,
                                                    *result._device);
    if (synchronize)
        cudaDeviceSynchronize();
    else
        return;
}

__device__ T *buffer;
int bufferCurrentSize = 0;
__global__ void AllocateBuffer(int size) {
    if (buffer)
        delete[] buffer;
    buffer = new T[size];
}

__global__ void DotK(D_Array &x, D_Array &y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= x.n)
        return;
    T safVal = x.vals[i];
    x.vals[i] = y.vals[i] * safVal;
    __syncthreads();
    for (int exp = 0; (1 << exp) < blockDim.x; exp++) {
        if (threadIdx.x % (2 << exp) == 0 &&
            threadIdx.x + (1 << exp) < blockDim.x) {
            x.vals[i] += x.vals[i + (1 << exp)];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        buffer[blockIdx.x] = x.vals[i];
    x.vals[i] = safVal;
    return;
}

__global__ void SumBlocks(T &result, int nValues) {
    result = 0;
    for (int b = 0; b < nValues; b++)
        result += buffer[b];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nValues)
        return;
    for (int exp = 0; (1 << exp) < blockDim.x; exp++) {
        if (threadIdx.x % (2 << exp) == 0 &&
            threadIdx.x + (1 << exp) < blockDim.x) {
            buffer[i] += buffer[i + (1 << exp)];
        }
        __syncthreads();
    }
    if (i == 0)
        result = buffer[i];
}

void Dot(D_Array &x, D_Array &y, T &result, bool synchronize) {
    assert(x.isDevice && y.isDevice);
    assert(x.n == y.n);
    dim3Pair threadblock = Make1DThreadBlock(x.n);
    if (bufferCurrentSize < threadblock.block.x) {
        AllocateBuffer<<<1, 1>>>(threadblock.block.x);
        bufferCurrentSize = threadblock.block.x;
    }
    DotK<<<threadblock.block, threadblock.thread>>>(*x._device, *y._device);
    cudaDeviceSynchronize();
    do {
        int nValues = threadblock.block.x;
        threadblock.block.x =
            int((threadblock.block.x - 1) / threadblock.thread.x) + 1;
        SumBlocks<<<threadblock.block.x, threadblock.thread.x>>>(result,
                                                                 nValues);
        cudaDeviceSynchronize();
    } while (threadblock.block.x > 1);
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
    // if (i >= a.i_size)
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
    assert(a.i_size == b.i_size && a.j_size == b.j_size);
    c.i_size = 1 * a.i_size;
    c.j_size = 1 * a.j_size;
    HDData<int> nnz;
    SumNNZK<<<1, 1>>>(*a._device, *b._device, nnz(true));
    cudaDeviceSynchronize();
    nnz.SetHost();
    c.SetNNZ(nnz());
    // dim3Pair threadblock = Make1DThreadBlock(a.i_size);
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