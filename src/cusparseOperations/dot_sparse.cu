#include "cuda_runtime.h"
#include <assert.h>
#include <stdio.h>

#include "cudaHelper/cuda_error_check.h"
#include "dot_sparse.hpp"
#include "sparseDataStruct/double_data.hpp"
#include "sparseDataStruct/matrix_element.hpp"

__global__ void DotK(MatrixSparse &d_mat, VectorDense &x, VectorDense &y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= d_mat.i_size)
        return;
    MatrixElement it(d_mat.rowPtr[i], &d_mat);
    y.vals[i] = 0;
    do {
        y.vals[i] += it.val[0] * x.vals[it.j];
        it.Next();
    } while (it.i == i);
}

void Dot(MatrixSparse &d_mat, VectorDense &x, VectorDense &result,
         bool synchronize) {
    assert(d_mat.isDevice && x.isDevice && result.isDevice);
    dim3 threadCount(BLOCK_SIZE * BLOCK_SIZE);
    dim3 blockCount(1);
    if (threadCount.x < d_mat.i_size) {
        blockCount.x = int((d_mat.i_size - 1) / threadCount.x) + 1;
    } else {
        threadCount.x = d_mat.i_size;
    }
    DotK<<<blockCount, threadCount>>>(*d_mat._device, *x._device,
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

__global__ void DotK(VectorDense &x, VectorDense &y) {
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

void Dot(VectorDense &x, VectorDense &y, double &result, bool synchronize) {
    assert(x.isDevice && y.isDevice);
    assert(x.n == y.n);
    dim3 threadCount(BLOCK_SIZE * BLOCK_SIZE);
    dim3 blockCount(1);
    if (threadCount.x < x.n) {
        blockCount.x = int((x.n - 1) / threadCount.x) + 1;
    } else {
        threadCount.x = x.n;
    }
    if (bufferCurrentSize < blockCount.x) {
        AllocateBuffer<<<1, 1>>>(blockCount.x);
        bufferCurrentSize = blockCount.x;
    }
    DotK<<<blockCount, threadCount>>>(*x._device, *y._device);
    cudaDeviceSynchronize();
    do {
        int nValues = blockCount.x;
        blockCount.x = int((blockCount.x - 1) / threadCount.x) + 1;
        SumBlocks<<<blockCount.x, threadCount.x>>>(result, nValues);
        cudaDeviceSynchronize();
    } while (blockCount.x > 1);
    if (synchronize)
        cudaDeviceSynchronize();
    else
        return;
}

__global__ void VectorSumK(VectorDense &a, VectorDense &b, T &alpha,
                           VectorDense &c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= a.n)
        return;
    c.vals[i] = a.vals[i] + b.vals[i] * alpha;
};

void VectorSum(VectorDense &a, VectorDense &b, T &alpha, VectorDense &c,
               bool synchronize) {
    assert(a.isDevice && b.isDevice);
    assert(a.n == b.n);
    dim3 threadCount(BLOCK_SIZE * BLOCK_SIZE);
    dim3 blockCount(1);
    if (threadCount.x < a.n) {
        blockCount.x = int((a.n - 1) / threadCount.x) + 1;
    } else {
        threadCount.x = a.n;
    }
    VectorSumK<<<blockCount, threadCount>>>(*a._device, *b._device, alpha,
                                            *c._device);
    if (synchronize)
        cudaDeviceSynchronize();
}

void VectorSum(VectorDense &a, VectorDense &b, VectorDense &c,
               bool synchronize) {
    HDData<T> p1(1.0);
    VectorSum(a, b, p1(true), c, synchronize);
}