#include "cuda_reduction_operation.hpp"

__global__ void ReductionIncreasingK(int *A, int n, int shift) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n)
        return;
    if (i - shift >= 0) {
        T val = A[i - shift];
        __syncthreads();
        A[i] += val;
    }
};

__global__ void check(int *A, int n) {
    for (int i = 0; i < n - 1; i++)
        if (A[i] >= A[i + 1])
            printf("%i  \n", i);
};

T ReductionIncreasing(int *A, int n) {
    dim3Pair threadblock;
    int shift = 1;
    do {
        threadblock = Make1DThreadBlock(n);
        ReductionIncreasingK<<<threadblock.block.x, threadblock.thread.x>>>(
            A, n, shift);
        cudaDeviceSynchronize();
        shift *= 2;
    } while (shift < n);
    check<<<1, 1>>>(A, n);
    return 0;
}