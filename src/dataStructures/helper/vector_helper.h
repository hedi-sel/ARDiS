#include "cuda_runtime.h"
#include "dataStructures/array.hpp"

template <typename C>
__device__ __host__ inline void printVectorBody(const D_Vector<C> &vector,
                                                int printCount) {
    printf("[ ");
    for (int i = 0; i < vector.n - 1 && i < printCount; i++)
        printf("%.3e, ", vector.vals[i]);
    if (printCount < vector.n - 1)
        printf("... ");
    printf("%.3e]\n", vector.vals[vector.n - 1]);
}

__global__ void printVector(const D_Vector<T> &vector, int printCount) {
    printVectorBody(vector, printCount);
}

__global__ void printVector(const D_Vector<int> &vector, int printCount) {
    printf("Cannont print int yet\n");
}
__global__ void printVector(const D_Vector<D_Array *> &vector, int printCount) {
    printf("Cannont print D_Array* yet\n");
}