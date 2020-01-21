#include "cuda_runtime.h"
#include "sparseDataStruct/vector_dense.hpp"

__device__ __host__ inline void printVectorBody(const D_Array &vector) {
    for (int i = 0; i < vector.n - 1; i++)
        printf("%.1f, ", vector.vals[i]);
    printf("%.1f]\n", vector.vals[vector.n - 1]);
}

__global__ void printVector(const D_Array &vector) {
    printVectorBody(vector);
}