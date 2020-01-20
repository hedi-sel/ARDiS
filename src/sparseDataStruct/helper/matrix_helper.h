#include "cuda_runtime.h"
#include "sparseDataStruct/matrix_element.hpp"
#include "sparseDataStruct/matrix_sparse.hpp"

__device__ __host__ inline void convertArrayBody(int *toOrderArray,
                                                 int nnz, int *newArray,
                                                 int newSize) {
    newArray[0] = 0;
    int it = 0;
    for (int k = 1; k < newSize; k++) {
        newArray[k] = newArray[k - 1];
        while (it < nnz && toOrderArray[it] == k - 1) {
            newArray[k]++;
            it++;
        }
    }
}

__global__ void convertArray(int *toOrderArray, int nnz, int *newArray,
                             int newSize) {
    convertArrayBody(toOrderArray, nnz, newArray, newSize);
}

__device__ __host__ void checkOrderedBody(int *array, int size, bool *_isOK) {
    bool isOK = true;
    for (int k = 1; k < size && isOK; k++) {
        isOK = isOK && array[k] >= array[k - 1];
    }
    *_isOK = isOK;
};

__global__ void checkOrdered(int *array, int size, bool *_isOK) {
    checkOrderedBody(array, size, _isOK);
}

__device__ __host__ inline void printMatrixBody(const MatrixSparse *matrix,
                                                int printCount = 0) {
    printf("Matrix :\n%i %i %i isDev=%i format=", matrix->i_size,
           matrix->j_size, matrix->nnz, matrix->isDevice);
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
        if (printCount > 0 && elm.k >= printCount) {
            printf("... \n");
            return;
        }
    }
}

__global__ void printMatrix(const MatrixSparse *matrix, int printCount) {
    printMatrixBody(matrix, printCount);
}

__device__ __host__ void IsSymetricBody(const MatrixSparse *matrix,
                                        bool *_return) {
    *_return = true;
    for (MatrixElement elm(matrix); elm.HasNext(); elm.Next()) {
        if (elm.i != elm.j && matrix->Lookup(elm.j, elm.i) != *elm.val) {
            printf("%i %i %f %f\n", elm.i, elm.j, *elm.val, 0.1);
            *_return = false;
            return;
        }
    }
    return;
}

__global__ void IsSymetricKernel(const MatrixSparse *matrix, bool *_return) {
    IsSymetricBody(matrix, _return);
}