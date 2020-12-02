#include "cuda_runtime.h"
#include "dataStructures/array.hpp"

__device__ __host__ inline void printVectorBody(const d_array<T> &vector,
                                                int printCount) {
    printf("[ ");
    for (int i = 0; i < vector.n - 1 && i < printCount; i++)
        printf("%.3e, ", vector.data[i]);
    if (printCount < vector.n - 1)
        printf("... ");
    printf("%.3e]\n", vector.data[vector.n - 1]);
}

__device__ __host__ void printVectorBody(const d_array<int> &vector,
                                         int printCount) {
    printf("[ ");
    for (int i = 0; i < vector.n - 1 && i < printCount; i++)
        printf("%i, ", vector.data[i]);
    if (printCount < vector.n - 1)
        printf("... ");
    printf("%i]\n", vector.data[vector.n - 1]);
}

__device__ __host__ void printVectorBody(const d_array<bool> &vector,
                                         int printCount) {
    printf("Printing d_array<bool> has not been implemented\n");
}

__device__ __host__ void printVectorBody(const d_array<d_vector *> &vector,
                                         int printCount) {
    printf("Printing d_array<d_vector *> has not been implemented\n");
}

// template <typename C>
// __global__ void printVectorK(const d_array<C> &vector, int printCount) {
//     printVectorBody(vector, printCount);
// }
__global__ void printVectorK(const d_array<T> &vector, int printCount) {
    printVectorBody(vector, printCount);
}
__global__ void printVectorK(const d_array<int> &vector, int printCount) {
    printVectorBody(vector, printCount);
}
__global__ void printVectorK(const d_array<bool> &vector, int printCount) {
    printVectorBody(vector, printCount);
}
__global__ void printVectorK(const d_array<d_vector *> &vector,
                             int printCount) {
    printVectorBody(vector, printCount);
}