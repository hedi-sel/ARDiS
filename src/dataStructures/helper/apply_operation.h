#include <nvfunctional>

#include "dataStructures/array.hpp"

// typedef nvstd::function<T &> Apply;
// template <typename T1, typename T2> __global__ void inserter(T1 *f, T2 l) {
//     *f = l;
// }
// typedef void (*FunctionDev)(...);

template <typename Apply>
__global__ void ApplyFunctionK(D_Array &vector, Apply func) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= vector.n)
        return;
    (func)(vector.vals[i]);
    return;
}

template <typename Apply>
__host__ void ApplyFunction(D_Array &vector, Apply func) {
    auto tb = Make1DThreadBlock(vector.n);
    ApplyFunctionK<<<tb.block, tb.thread>>>(*vector._device, func);
};

template <typename Apply>
__global__ void ApplyFunctionConditionalK(D_Array &vector, D_Array &booleans,
                                          Apply func) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= vector.n)
        return;
    if (booleans.vals[i] > 0)
        (func)(vector.vals[i]);
    return;
}

template <typename Apply>
__host__ void ApplyFunctionConditional(D_Array &vector, D_Array &booleans,
                                       Apply func) {
    auto tb = Make1DThreadBlock(vector.n);
    ApplyFunctionConditionalK<<<tb.block, tb.thread>>>(*vector._device,
                                                       *booleans._device, func);
};

template <typename Reduction>
__global__ void ReductionFunctionK(D_Array &A, int nValues, int shift,
                                   Reduction func) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nValues)
        return;
    for (int exp = 0; (1 << exp) < blockDim.x; exp++) {
        if (threadIdx.x % (2 << exp) == 0 &&
            threadIdx.x + (1 << exp) < blockDim.x &&
            i + (1 << exp) <= nValues) {
            A.vals[shift * i] =
                func(A.vals[shift * i], A.vals[shift * (i + (1 << exp))]);
        }
        __syncthreads();
    }
};

template <typename Reduction> T ReductionFunction(D_Array &A, Reduction func) {
    int nValues = A.n;
    dim3Pair threadblock;
    int shift = 1;
    do {
        threadblock = Make1DThreadBlock(nValues);
        ReductionK<<<threadblock.block.x, threadblock.thread.x>>>(
            *A._device, nValues, shift, func);
        gpuErrchk(cudaDeviceSynchronize());
        nValues = int((nValues - 1) / threadblock.thread.x) + 1;
        shift *= threadblock.thread.x;
    } while (nValues > 1);
    return 0;
}

template <typename Reduction>
__global__ void ReductionFunctionConditionalK(D_Array &A, D_Array &booleans,
                                              int nValues, int shift,
                                              Reduction func) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nValues)
        return;
    for (int exp = 0; (1 << exp) < blockDim.x; exp++) {
        if (threadIdx.x % (2 << exp) == 0 &&
            threadIdx.x + (1 << exp) < blockDim.x &&
            i + (1 << exp) <= nValues) {
            if (booleans.vals[shift * (i + (1 << exp))] >
                0) // TODO Make this as array of bool
                if (booleans.vals[shift * i] > 0)
                    A.vals[shift * i] = func(A.vals[shift * i],
                                             A.vals[shift * (i + (1 << exp))]);
                else
                    A.vals[shift * i] = A.vals[shift * (i + (1 << exp))];
            booleans.vals[shift * i] = booleans.vals[shift * i] +
                                       booleans.vals[shift * (i + (1 << exp))];
        }
        __syncthreads();
    }
};

template <typename Reduction>
T ReductionFunctionConditional(D_Array &A, D_Array &booleans, Reduction func) {
    int nValues = A.n;
    dim3Pair threadblock;
    int shift = 1;
    do {
        threadblock = Make1DThreadBlock(nValues);
        ReductionFunctionConditionalK<<<threadblock.block.x,
                                        threadblock.thread.x>>>(
            *A._device, *booleans._device, nValues, shift, func);
        gpuErrchk(cudaDeviceSynchronize());
        nValues = int((nValues - 1) / threadblock.thread.x) + 1;
        shift *= threadblock.thread.x;
    } while (nValues > 1);
    return 0;
}
