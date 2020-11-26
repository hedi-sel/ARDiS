#pragma once

#include <cstdarg>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <cusparse_v2.h>
#include <string>
#include <utility>

#include "constants.hpp"

enum MatrixType { COO, CSR, CSC };

class D_SparseMatrix {
  public:
    int nnz;
    int rows;
    int cols;

    int loaded_elements = 0;
    int dataWidth = -1;

    MatrixType type;
    const bool isDevice;
    // cusparseMatDescr_t descr = NULL;

    T *data;
    int *rowPtr;
    int *colPtr;
    D_SparseMatrix *_device;

    __host__ D_SparseMatrix();
    __host__ D_SparseMatrix(int rows, int cols, int nnz = 0, MatrixType = COO,
                            bool isDevice = true);
    __host__ D_SparseMatrix(const D_SparseMatrix &,
                            bool copyToOtherMem = false);
    __host__ void operator=(const D_SparseMatrix &);
    __host__ bool operator==(const D_SparseMatrix &);

    __host__ ~D_SparseMatrix();

    __host__ std::string ToString();
    __host__ __device__ void Print(int printCount = 5) const;

    __host__ void SetNNZ(int);

    // Add an element at index k
    __host__ void StartFilling();
    __host__ __device__ void AddElement(int i, int j, T);

    // __host__ __device__ MatrixElement Start() const;

    // Get the value at index k of the sparse matrix
    __host__ __device__ const T &Get(int k) const;
    __host__ __device__ const T &GetLine(int i) const;

    // Set the element at index k of the matrix
    __host__ __device__ void Set(int k, const T);
    // Get the element at position (i, j) in the matrix, if it is defined
    __host__ __device__ T Lookup(int i, int j) const;

    // Turn the matrix to CSC or CSR type
    __host__ void ToCompressedDataType(MatrixType = COO);
    __host__ bool IsConvertibleTo(MatrixType) const;

    __host__ void ConvertMatrixToCSR();

    __host__ bool IsSymetric();

    __host__ cusparseMatDescr_t MakeDescriptor();
    __host__ cusparseSpMatDescr_t MakeSpDescriptor();

    __host__ void OperationCuSparse(void *function, cusparseHandle_t &,
                                    bool addValues = false, void * = NULL,
                                    void * = NULL);
    __host__ void OperationCuSolver(void *function, cusolverSpHandle_t &,
                                    cusparseMatDescr_t, T *b = NULL,
                                    T *xOut = NULL, int *singularOut = NULL);

    __host__ void MakeDataWidth();

  private:
    __host__ void MemAlloc();
    __host__ void MemFree();
};