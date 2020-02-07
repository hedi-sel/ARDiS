#include "cuda_runtime.h"
#include "dataStructures/array.hpp"

__device__ __host__ inline void printVectorBody(const D_Array &vector,
                                                int printCount) {
    for (int i = 0; i < vector.n - 1 && i < printCount; i++)
        printf("%.3e, ", vector.vals[i]);
    if (printCount < vector.n - 1)
        printf("... ");
    printf("%.3e]\n", vector.vals[vector.n - 1]);
}

__global__ void printVector(const D_Array &vector, int printCount) {
    printVectorBody(vector, printCount);
}