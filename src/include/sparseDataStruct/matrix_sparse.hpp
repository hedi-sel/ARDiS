#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <fstream>
#include <cstdarg>

#include "constants.h"
#include "cudaHelper/cuda_error_check.h"
#include "cudaHelper/cusparse_error_check.h"
#include "matrix_element.hpp"

enum MatrixType
{
    COO,
    CSR
};

class MatrixSparse
{
public:
    const int n_elements;
    const int i_size;
    const int j_size;

    const MatrixType type;
    const bool isDevice;

    __host__ MatrixSparse(int i_size, int j_size, int n_elements, MatrixType = CSR, bool isDevice = false);
    __host__ MatrixSparse(const MatrixSparse &, bool copyToOtherMem = false);
    __host__ ~MatrixSparse();

    // Add an element at index k
    __host__ __device__ void AddElement(int k, int i, int j, const T);

    // Get the value at index k of the sparse matrix
    __host__ __device__ inline const MatrixElement Get(int k)
    {
        MatrixElement elm(rowPtr[k], colPtr[k], vals[k]);
        return elm;
    }
    // Set the element at index k of the matrix
    __host__ __device__ void Set(int k, const T);
    // Get the element at position (i, j) in the matrix, if it is defined
    __host__ __device__ const T &Get(int i, int j);

    __host__ void OperationCuSparse(void *operation, cusparseHandle_t &);
    __host__ void OperationCuSparse(void *operation, cusparseHandle_t &, size_t *pBufferSizeInBytes);
    __host__ void OperationCuSparse(void *operation, cusparseHandle_t &, void *pBuffer);

    __host__ __device__ void Print();

    T *vals;
    int *rowPtr;
    int *colPtr;
    MatrixSparse *_device;

private:
    __host__ void MemAlloc();
};