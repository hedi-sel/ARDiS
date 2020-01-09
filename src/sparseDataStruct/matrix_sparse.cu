#include <assert.h>

#include "cudaHelper/cuda_error_check.h"
#include "cudaHelper/cusolverSP_error_check.h"
#include "cudaHelper/cusparse_error_check.h"
#include "cusparseOperations/row_ordering.hpp"
#include "sparseDataStruct/helper/matrix_helper.h"
#include <sparseDataStruct/matrix_element.hpp>
#include <sparseDataStruct/matrix_sparse.hpp>

__host__ MatrixSparse::MatrixSparse(int i_size, int j_size, int n_elements,
                                    MatrixType type, bool isDevice)
    : n_elements(n_elements), i_size(i_size), j_size(j_size),
      isDevice(isDevice), type(type) {
    MemAlloc();
}

__host__ MatrixSparse::MatrixSparse(const MatrixSparse &m, bool copyToOtherMem)
    : MatrixSparse(m.i_size, m.j_size, m.n_elements, m.type,
                   m.isDevice ^ copyToOtherMem) {
    cudaMemcpyKind memCpy =
        (m.isDevice)
            ? (isDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost
            : (isDevice) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    gpuErrchk(cudaMemcpy(vals, m.vals, sizeof(T) * n_elements, memCpy));
    gpuErrchk(cudaMemcpy(
        colPtr, m.colPtr,
        sizeof(int) * ((type == CSC) ? j_size + 1 : n_elements), memCpy));
    gpuErrchk(cudaMemcpy(
        rowPtr, m.rowPtr,
        sizeof(int) * ((type == CSR) ? i_size + 1 : n_elements), memCpy));
}

__host__ void MatrixSparse::MemAlloc() {
    int rowPtrSize = (type == CSR) ? i_size + 1 : n_elements;
    int colPtrSize = (type == CSC) ? j_size + 1 : n_elements;
    if (isDevice) {
        gpuErrchk(cudaMalloc(&vals, n_elements * sizeof(T)));
        gpuErrchk(cudaMalloc(&rowPtr, rowPtrSize * sizeof(int)));
        gpuErrchk(cudaMalloc(&colPtr, colPtrSize * sizeof(int)));

        gpuErrchk(cudaMalloc(&_device, sizeof(MatrixSparse)));
        gpuErrchk(cudaMemcpy(_device, this, sizeof(MatrixSparse),
                             cudaMemcpyHostToDevice));
    } else {
        vals = new T[n_elements];
        rowPtr = new int[rowPtrSize];
        for (int i = 0; i < rowPtrSize; i++)
            rowPtr[i] = 0;
        colPtr = new int[colPtrSize];
        for (int i = 0; i < colPtrSize; i++)
            colPtr[i] = 0;
    }
}

__host__ __device__ void MatrixSparse::Print() const {
#ifndef __CUDA_ARCH__
    if (isDevice) {
        printMatrix<<<1, 1>>>(_device);
        cudaDeviceSynchronize();
    } else
#endif
        printMatrixBody(this);
}

__host__ __device__ void MatrixSparse::AddElement(int k, int i, int j,
                                                  const T val) {
    assert(!isDevice);
    vals[k] = val;
    if (type == CSC) {
        if (colPtr[j + 1] == 0)
            colPtr[j + 1] = colPtr[j];
        colPtr[j + 1]++;
    } else {
        colPtr[k] = j;
    }
    if (type == CSR) {
        if (rowPtr[i + 1] == 0)
            rowPtr[i + 1] = rowPtr[i];
        rowPtr[i + 1]++;
    } else {
        rowPtr[k] = i;
    }
}

// Get the value at index k of the sparse matrix
__host__ __device__ const T &MatrixSparse::Get(int k) const { return vals[k]; }
__host__ __device__ const T &MatrixSparse::GetLine(int i) const {
    if (type != CSR) {
        printf("Error! Doesn't work with other type than CSR");
    }
    return vals[rowPtr[i]];
}

__host__ __device__ T MatrixSparse::Lookup(int i, int j) const {
    for (MatrixElement elm(this); elm.HasNext(); elm.Next())
        if (elm.i == i && elm.j == j)
            return *elm.val;
    return 0;
}

__host__ void MatrixSparse::ToCompressedDataType(MatrixType toType,
                                                 bool orderBeforhand) {
    if (toType == COO) {
        if (IsConvertibleTo(CSR))
            toType = CSR;
        else if (IsConvertibleTo(CSC))
            toType = CSC;
        else {
            printf("Not convertible to any type!\n");
            return;
        }
    } else {
        assert(IsConvertibleTo(toType));
    }
    int newSize = (toType == CSR) ? i_size + 1 : j_size + 1;
    int *newArray;
    if (isDevice) {
        gpuErrchk(cudaMalloc(&newArray, newSize * sizeof(int)));
        convertArray<<<1, 1>>>((toType == CSR) ? rowPtr : colPtr, n_elements,
                               newArray, newSize);
        cudaFree((toType == CSR) ? rowPtr : colPtr);
    } else {
        newArray = new int[newSize];
        convertArrayBody((toType == CSR) ? rowPtr : colPtr, n_elements,
                         newArray, newSize);
        if (toType == CSR)
            delete[] rowPtr;
        else
            delete[] colPtr;
    }
    if (toType == CSR)
        rowPtr = newArray;
    else
        colPtr = newArray;
    type = toType;
    if (isDevice) {
        gpuErrchk(cudaMemcpy(_device, this, sizeof(MatrixSparse),
                             cudaMemcpyHostToDevice));
        gpuErrchk(cudaDeviceSynchronize());
    }
}

__host__ bool MatrixSparse::IsConvertibleTo(MatrixType toType) const {
    assert(toType != type);
    if (toType == COO)
        return true;
    if (type != COO)
        return false;
    int *analyzedArray = (toType == CSR) ? rowPtr : colPtr;
    bool isOK = true;
    if (isDevice) {
        bool *_isOK;
        gpuErrchk(cudaMalloc(&_isOK, sizeof(bool)));
        checkOrdered<<<1, 1>>>(analyzedArray, n_elements, _isOK);
        gpuErrchk(
            cudaMemcpy(&isOK, _isOK, sizeof(bool), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(_isOK));
        gpuErrchk(cudaDeviceSynchronize());
    } else {
        checkOrderedBody(analyzedArray, n_elements, &isOK);
    }
    return isOK;
}

__host__ void MatrixSparse::ConvertMatrixToCSR() {
    if (type == CSR)
        throw("Error! Already CSR type \n");
    if (type == CSC)
        throw("Error! Already CSC type \n");
    if (!IsConvertibleTo(CSR)) {
        RowOrdering(*this);
    }
    assert(IsConvertibleTo(CSR));
    ToCompressedDataType(CSR);
    assert(type == CSR);
}

__host__ void MatrixSparse::MakeDescriptor() {
    if (descr == NULL) {
        cusparseErrchk(cusparseCreateMatDescr(&descr));
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    } else {
        printf("Matrix already has a descriptor!");
    }
}

__host__ bool MatrixSparse::IsSymetric() {
    bool *_return = new bool;
    if (isDevice) {
        bool *_returnGpu;
        gpuErrchk(cudaMalloc(&_returnGpu, sizeof(bool)));
        IsSymetricKernel<<<1, 1>>>(_device, _returnGpu);
        cudaDeviceSynchronize();
        gpuErrchk(cudaMemcpy(_return, _returnGpu, sizeof(bool),
                             cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(_returnGpu));
        gpuErrchk(cudaDeviceSynchronize());
    } else {
        IsSymetricBody(this, _return);
    }
    return *_return;
}

typedef cusparseStatus_t (*FuncSpar)(...);
__host__ void MatrixSparse::OperationCuSparse(void *function,
                                              cusparseHandle_t &handle,
                                              bool addValues, void *pointer1,
                                              void *pointer2) {
    if (addValues) {
        printf("This function is not complete");
    } else {
        if (pointer1)
            if (pointer2) {
                cusparseErrchk(((FuncSpar)function)(handle, i_size, j_size,
                                                    n_elements, rowPtr, colPtr,
                                                    pointer1, pointer2));
            } else {
                cusparseErrchk(((FuncSpar)function)(handle, i_size, j_size,
                                                    n_elements, rowPtr, colPtr,
                                                    pointer1));
            }
        else
            printf("This function is not complete");
    }
}

typedef cusolverStatus_t (*FuncSolv)(...);
__host__ void MatrixSparse::OperationCuSolver(void *function,
                                              cusolverSpHandle_t &handle, T *b,
                                              T *xOut, int *singularOut) {
    cusolverErrchk(((FuncSolv)function)(handle, i_size, n_elements, descr, vals,
                                        rowPtr, colPtr, b, 0.0, 0, xOut,
                                        singularOut));
    // TODO : SymOptimization
}

__host__ MatrixSparse::~MatrixSparse() {
    if (isDevice) {
        gpuErrchk(cudaFree(vals));
        gpuErrchk(cudaFree(rowPtr));
        gpuErrchk(cudaFree(colPtr));
        gpuErrchk(cudaFree(_device));
    } else {
        delete[] vals;
        delete[] rowPtr;
        delete[] colPtr;
    }
}
