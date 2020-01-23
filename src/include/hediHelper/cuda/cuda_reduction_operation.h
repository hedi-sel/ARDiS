#pragma once

#include <cuda_runtime.h>

#include "constants.hpp"
#include "dataStructures/array.hpp"
#include "dataStructures/hd_data.hpp"
#include "hediHelper/cuda/cuda_thread_manager.hpp"

template <typename OpType>
__global__ void ReductionK(D_Array &A, int nValues, int shift, OpType op) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nValues)
        return;
    for (int exp = 0; (1 << exp) < blockDim.x; exp++) {
        if (threadIdx.x % (2 << exp) == 0 &&
            threadIdx.x + (1 << exp) < blockDim.x) {
            A.vals[i] = op(A.vals[i], A.vals[i + shift * (1 << exp)]);
        }
        __syncthreads();
        // if (i == 0)
        //     A.Print();
    }
};

template <typename OpType> T ReductionOperation(D_Array &A, OpType op) {
    int nValues = A.n;
    dim3Pair threadblock;
    int shift = 1;
    do {
        threadblock = Make1DThreadBlock(nValues);
        ReductionK<<<threadblock.block.x, threadblock.thread.x>>>(
            *A._device, nValues, shift, op);
        nValues = int((nValues - 1) / threadblock.thread.x) + 1;
        shift *= threadblock.thread.x;
    } while (nValues > 1);
    // A.Print();
    return 0;
}
// ReductionOperation<>;