#include "cuda_runtime.h"
#include "dataStructures/array.hpp"

template <typename C>
__device__ __host__ inline void printVectorBody(const D_Array<C> &vector,
                                                int printCount) {
    printf("[ ");
    for (int i = 0; i < vector.n - 1 && i < printCount; i++)
        printf("%.3e, ", vector.data[i]);
    if (printCount < vector.n - 1)
        printf("... ");
    printf("%.3e]\n", vector.data[vector.n - 1]);
}

__global__ void printVector(const D_Array<T> &vector, int printCount) {
    printVectorBody(vector, printCount);
}

__global__ void printVector(const D_Array<int> &vector, int printCount) {
    printf("Printing D_Array<int> has not been implemented\n");
}
__global__ void printVector(const D_Array<D_Vector *> &vector, int printCount) {
    printf("Printing D_Array<D_Vector *> has not been implemented\n");
}
__global__ void printVector(const D_Array<bool> &vector, int printCount) {
    printf("Printing D_Array<bool> has not been implemented\n");
}