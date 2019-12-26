#include "cuda_runtime.h"
#include "sparseDataStruct/matrix_element.hpp"
#include "sparseDataStruct/matrix_sparse.hpp"

__device__ __host__ inline void convertArrayBody(int *toOrderArray,
                                                 int n_elements, int *newArray,
                                                 int newSize) {
    newArray[0] = 0;
    int it = 0;
    for (int k = 1; k < newSize; k++) {
        newArray[k] = newArray[k - 1];
        while (it < n_elements && toOrderArray[it] == k - 1) {
            newArray[k]++;
            it++;
        }
    }
}

__global__ void convertArray(int *toOrderArray, int n_elements, int *newArray,
                             int newSize) {
    convertArrayBody(toOrderArray, n_elements, newArray, newSize);
}

__global__ void checkOrdered(int *array, int size, bool *_isOK) {
    bool isOK = true;
    for (int k = 1; k < size && isOK; k++) {
        isOK = isOK && array[k] >= array[k - 1];
    }
    *_isOK = isOK;
}

__device__ __host__ inline void printMatrixBody(const MatrixSparse *matrix) {
    MatrixElement elm(matrix);
    printf("Matrix :\n%i %i %i format=", matrix->i_size, matrix->j_size,
           matrix->n_elements);
    switch (matrix->type) {
    case COO:
        printf("COO\n");
        break;
    case CSR:
        printf("CSR\n");
        break;
    case CSC:
        printf("CSC\n");
        break;
    }
    for (MatrixElement elm(matrix); elm.HasNext(); elm.Next()) {
        elm.Print();
    }
}

__global__ void printMatrix(const MatrixSparse *matrix) {
    printMatrixBody(matrix);
}