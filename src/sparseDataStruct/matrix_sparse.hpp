#pragma once

#include <cstdarg>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <utility>

#include "constants.hpp"

enum MatrixType { COO, CSR, CSC };

class MatrixSparse {
  public:
    int n_elements;
    int i_size;
    int j_size;

    MatrixType type;
    const bool isDevice;
    cusparseMatDescr_t descr = NULL;

    T *vals;
    int *rowPtr;
    int *colPtr;
    MatrixSparse *_device;

    __host__ MatrixSparse(int i_size, int j_size, int n_elements,
                          MatrixType = CSR, bool isDevice = false);
    __host__ MatrixSparse(const MatrixSparse &, bool copyToOtherMem = false);
    __host__ ~MatrixSparse();

    __host__ __device__ void Print() const;

    // Add an element at index k
    __host__ __device__ void AddElement(const int k, const int i, const int j,
                                        const T);

    // __host__ __device__ MatrixElement Start() const;

    // Get the value at index k of the sparse matrix
    __host__ __device__ const T &Get(int k) const;
    __host__ __device__ const T &GetLine(int i) const;

    // Set the element at index k of the matrix
    __host__ __device__ void Set(int k, const T);
    // Get the element at position (i, j) in the matrix, if it is defined
    __host__ __device__ T Lookup(int i, int j) const;

    // Turn the matrix to CSC or CSR type
    __host__ void ToCompressedDataType(MatrixType = COO,
                                       bool orderBeforhand = false);
    __host__ bool IsConvertibleTo(MatrixType) const;

    __host__ void ConvertMatrixToCSR();

    __host__ bool IsSymetric();

    __host__ void MakeDescriptor();
    __host__ void OperationCuSparse(void *function, cusparseHandle_t &,
                                    bool addValues = false, void * = NULL,
                                    void * = NULL);
    __host__ void OperationCuSolver(void *function, cusolverSpHandle_t &,
                                    T *b = NULL, T *xOut = NULL,
                                    int *singularOut = NULL);

  private:
    __host__ void MemAlloc();
};