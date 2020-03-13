#include "cuda_reduction_operation.hpp"

__global__ void ReductionK(D_Array &A, int nValues, int shift, int op) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nValues)
        return;
    for (int exp = 0; (1 << exp) < blockDim.x; exp++) {
        if (threadIdx.x % (2 << exp) == 0 &&
            threadIdx.x + (1 << exp) < blockDim.x && i + (1 << exp) < nValues) {
            if (op == sum)
                A.vals[shift * i] =
                    A.vals[shift * i] + A.vals[shift * (i + (1 << exp))];
            else if (op == maximum)
                A.vals[shift * i] =
                    (A.vals[shift * i] > A.vals[shift * (i + (1 << exp))])
                        ? A.vals[shift * i]
                        : A.vals[shift * (i + (1 << exp))];
        }
        __syncthreads();
    }
};

T ReductionOperation(D_Array &A, OpType op) {
    int nValues = A.n;
    dim3Pair threadblock;
    int shift = 1;
    do {
        threadblock = Make1DThreadBlock(nValues);
        ReductionK<<<threadblock.block.x, threadblock.thread.x>>>(
            *(D_Array *)A._device, nValues, shift, static_cast<int>(op));
        gpuErrchk(cudaDeviceSynchronize());
        nValues = int((nValues - 1) / threadblock.thread.x) + 1;
        shift *= threadblock.thread.x;
    } while (nValues > 1);
    return 0;
}

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