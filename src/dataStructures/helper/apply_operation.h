#include <nvfunctional>

#include "dataStructures/array.hpp"

// typedef nvstd::function<T &> apply;
// template <typename T1, typename T2> __global__ void inserter(T1 *f, T2 l) {
//     *f = l;
// }
// typedef void (*FunctionDev)(...);

template <typename apply, typename C>
__global__ void apply_functionK(d_array<C> &vector, apply func) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= vector.n)
        return;
    (func)(vector.data[i]);
    return;
}

template <typename apply, typename C>
__host__ void apply_func(d_array<C> &vector, apply func) {
    auto tb = make1DThreadBlock(vector.n);
    apply_functionK<<<tb.block, tb.thread>>>(*vector._device, func);
};

template <typename apply, typename C>
__global__ void apply_func_condK(d_array<C> &vector, d_array<bool> &booleans,
                                 apply func) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= vector.n)
        return;
    if (booleans.data[i])
        (func)(vector.data[i]);
    return;
}

template <typename apply, typename C>
__host__ void apply_func_cond(d_array<C> &vector, d_array<bool> &booleans,
                              apply func) {
    auto tb = make1DThreadBlock(vector.n);
    apply_func_condK<<<tb.block, tb.thread>>>(*vector._device,
                                              *booleans._device, func);
};

template <typename Reduction, typename C>
__global__ void reduction_funcK(d_array<C> &A, int nValues, int shift,
                                Reduction func) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nValues)
        return;
    for (int exp = 0; (1 << exp) < blockDim.x; exp++) {
        if (threadIdx.x % (2 << exp) == 0 &&
            threadIdx.x + (1 << exp) < blockDim.x && i + (1 << exp) < nValues) {
            A.data[shift * i] =
                func(A.data[shift * i], A.data[shift * (i + (1 << exp))]);
        }
        __syncthreads();
    }
};

template <typename Reduction, typename C>
C reduction_func(d_array<C> &A, Reduction func) {
    int nValues = A.n;
    dim3Pair threadblock;
    int shift = 1;
    do {
        threadblock = make1DThreadBlock(nValues);
        ReductionK<<<threadblock.block.x, threadblock.thread.x>>>(
            *A._device, nValues, shift, func);
        gpuErrchk(cudaDeviceSynchronize());
        nValues = int((nValues - 1) / threadblock.thread.x) + 1;
        shift *= threadblock.thread.x;
    } while (nValues > 1);
    return 0;
}

template <typename Reduction, typename C>
__global__ void reduction_func_condK(d_array<C> &A, d_array<bool> &booleans,
                                     int nValues, int shift, Reduction func) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nValues)
        return;
    for (int exp = 0; (1 << exp) < blockDim.x; exp++) {
        if (threadIdx.x % (2 << exp) == 0 &&
            threadIdx.x + (1 << exp) < blockDim.x && i + (1 << exp) < nValues) {
            if (booleans.data[shift * (i + (1 << exp))] >
                0) // TODO Make this as array of bool
                if (booleans.data[shift * i])
                    A.data[shift * i] = func(A.data[shift * i],
                                             A.data[shift * (i + (1 << exp))]);
                else
                    A.data[shift * i] = A.data[shift * (i + (1 << exp))];
            booleans.data[shift * i] = booleans.data[shift * i] ||
                                       booleans.data[shift * (i + (1 << exp))];
        }
        __syncthreads();
    }
};

template <typename Reduction, typename C>
C reduction_func_cond(d_array<C> &A, d_array<bool> &booleans, Reduction func) {
    int nValues = A.n;
    dim3Pair threadblock;
    int shift = 1;
    do {
        threadblock = make1DThreadBlock(nValues);
        reduction_func_condK<<<threadblock.block.x, threadblock.thread.x>>>(
            *A._device, *booleans._device, nValues, shift, func);
        gpuErrchk(cudaDeviceSynchronize());
        nValues = int((nValues - 1) / threadblock.thread.x) + 1;
        shift *= threadblock.thread.x;
    } while (nValues > 1);
    return 0;
}
