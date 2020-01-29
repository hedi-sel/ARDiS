#include "cuda_reduction_operation.hpp"

__device__ int *bufferRed;
int bufferRedSize = 0;
__global__ void AllocateBuffer(int size) {
    if (bufferRed)
        delete[] bufferRed;
    bufferRed = new int[size];
}

__global__ void ReductionIncreasing1K(int *A, int n, int shift) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n)
        return;
    if (i - shift >= 0) {
        bufferRed[i] = A[i] + A[i - shift];
    }
};

__global__ void ReductionIncreasing2K(int *A, int n, int shift) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n)
        return;
    if (i - shift >= 0) {
        A[i] = bufferRed[i];
    }
};

T ReductionIncreasing(int *A, int n) {
    if (bufferRedSize < n) {
        AllocateBuffer<<<1, 1>>>(n);
        bufferRedSize = n;
    }
    dim3Pair threadblock;
    int shift = 1;
    do {
        threadblock = Make1DThreadBlock(n);
        ReductionIncreasing1K<<<threadblock.block.x, threadblock.thread.x>>>(
            A, n, shift);
        cudaDeviceSynchronize();
        ReductionIncreasing2K<<<threadblock.block.x, threadblock.thread.x>>>(
            A, n, shift);
        cudaDeviceSynchronize();
        shift *= 2;
    } while (shift < n);
    return 0;
}