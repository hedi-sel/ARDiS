#include <assert.h>

#include "cudaHelper/cuda_error_check.h"
#include "cudaHelper/cusolverSP_error_check.h"
#include "cudaHelper/cusparse_error_check.h"
#include "cusparseOperations/row_ordering.hpp"
#include "sparseDataStruct/helper/matrix_helper.h"
#include <sparseDataStruct/matrix_element.hpp>
#include <sparseDataStruct/matrix_sparse.hpp>

__host__ D_SparseMatrix::D_SparseMatrix() : D_SparseMatrix(0, 0){};

__host__ D_SparseMatrix::D_SparseMatrix(int i_size, int j_size, int nnz,
                                        MatrixType type, bool isDevice)
    : nnz(nnz), i_size(i_size), j_size(j_size), isDevice(isDevice), type(type) {
    MemAlloc();
}

__host__ D_SparseMatrix::D_SparseMatrix(const D_SparseMatrix &m,
                                        bool copyToOtherMem)
    : D_SparseMatrix(m.i_size, m.j_size, m.nnz, m.type,
                     m.isDevice ^ copyToOtherMem) {
    loaded_elements = m.loaded_elements;
    assert(m.loaded_elements == m.nnz);
    cudaMemcpyKind memCpy =
        (m.isDevice)
            ? (isDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost
            : (isDevice) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    gpuErrchk(cudaMemcpy(vals, m.vals, sizeof(T) * nnz, memCpy));
    gpuErrchk(cudaMemcpy(colPtr, m.colPtr,
                         sizeof(int) * ((type == CSC) ? j_size + 1 : nnz),
                         memCpy));
    gpuErrchk(cudaMemcpy(rowPtr, m.rowPtr,
                         sizeof(int) * ((type == CSR) ? i_size + 1 : nnz),
                         memCpy));
}

__host__ void D_SparseMatrix::MemAlloc() {
    if (nnz == 0)
        return;
    int rowPtrSize = (type == CSR) ? i_size + 1 : nnz;
    int colPtrSize = (type == CSC) ? j_size + 1 : nnz;
    if (isDevice) {
        gpuErrchk(cudaMalloc(&vals, nnz * sizeof(T)));
        gpuErrchk(cudaMalloc(&rowPtr, rowPtrSize * sizeof(int)));
        gpuErrchk(cudaMalloc(&colPtr, colPtrSize * sizeof(int)));
        gpuErrchk(cudaMalloc(&_device, sizeof(D_SparseMatrix)));
        gpuErrchk(cudaMemcpy(_device, this, sizeof(D_SparseMatrix),
                             cudaMemcpyHostToDevice));
    } else {
        vals = new T[nnz];
        rowPtr = new int[rowPtrSize];
        for (int i = 0; i < rowPtrSize; i++)
            rowPtr[i] = 0;
        colPtr = new int[colPtrSize];
        for (int i = 0; i < colPtrSize; i++)
            colPtr[i] = 0;
    }
}

__host__ __device__ void D_SparseMatrix::Print(int printCount) const {
#ifndef __CUDA_ARCH__
    if (isDevice) {
        printMatrix<<<1, 1>>>(_device, printCount);
        cudaDeviceSynchronize();
    } else
#endif
        printMatrixBody(this, printCount);
}

__host__ void D_SparseMatrix::SetNNZ(int nnz) {
    assert(this->nnz == 0);
    this->nnz = nnz;
    MemAlloc();
}

__host__ __device__ void D_SparseMatrix::AddElement(int i, int j, T &val) {
#ifndef __CUDA_ARCH__
    if (isDevice) {
        AddElementK<<<1, 1>>>(_device, i, j, val);
        cudaDeviceSynchronize();
    } else
#endif
        AddElementBody(this, i, j, val);
}

// Get the value at index k of the sparse matrix
__host__ __device__ const T &D_SparseMatrix::Get(int k) const {
    return vals[k];
}
__host__ __device__ const T &D_SparseMatrix::GetLine(int i) const {
    if (type != CSR) {
        printf("Error! Doesn't work with other type than CSR");
    }
    return vals[rowPtr[i]];
}

__host__ __device__ T D_SparseMatrix::Lookup(int i, int j) const {
    for (MatrixElement elm(this); elm.HasNext(); elm.Next())
        if (elm.i == i && elm.j == j)
            return *elm.val;
    return 0;
}

__host__ void D_SparseMatrix::ToCompressedDataType(MatrixType toType) {
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
        convertArray<<<1, 1>>>(_device, (toType == CSR) ? rowPtr : colPtr,
                               newArray, newSize);
        cudaFree((toType == CSR) ? rowPtr : colPtr);
    } else {
        newArray = new int[newSize];
        convertArrayBody(this, (toType == CSR) ? rowPtr : colPtr, newArray,
                         newSize);
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
        loaded_elements = nnz; // Warning!! There is no assert to protect this!
        gpuErrchk(cudaMemcpy(_device, this, sizeof(D_SparseMatrix),
                             cudaMemcpyHostToDevice));
        gpuErrchk(cudaDeviceSynchronize());
    }
}

__host__ bool D_SparseMatrix::IsConvertibleTo(MatrixType toType) const {
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
        checkOrdered<<<1, 1>>>(analyzedArray, nnz, _isOK);
        gpuErrchk(
            cudaMemcpy(&isOK, _isOK, sizeof(bool), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(_isOK));
        gpuErrchk(cudaDeviceSynchronize());
    } else {
        checkOrderedBody(analyzedArray, nnz, &isOK);
    }
    return isOK;
}

__host__ void D_SparseMatrix::ConvertMatrixToCSR() {
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

__host__ void D_SparseMatrix::MakeDescriptor() {
    if (descr == NULL) {
        cusparseErrchk(cusparseCreateMatDescr(&descr));
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    } else {
        printf("Matrix already has a descriptor!");
    }
}

__host__ bool D_SparseMatrix::IsSymetric() {
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
__host__ void D_SparseMatrix::OperationCuSparse(void *function,
                                                cusparseHandle_t &handle,
                                                bool addValues, void *pointer1,
                                                void *pointer2) {
    if (addValues) {
        printf("This function is not complete");
    } else {
        if (pointer1)
            if (pointer2) {
                cusparseErrchk(((FuncSpar)function)(handle, i_size, j_size, nnz,
                                                    rowPtr, colPtr, pointer1,
                                                    pointer2));
            } else {
                cusparseErrchk(((FuncSpar)function)(handle, i_size, j_size, nnz,
                                                    rowPtr, colPtr, pointer1));
            }
        else
            printf("This function is not complete");
    }
}

typedef cusolverStatus_t (*FuncSolv)(...);
__host__ void D_SparseMatrix::OperationCuSolver(void *function,
                                                cusolverSpHandle_t &handle,
                                                T *b, T *xOut,
                                                int *singularOut) {
    cusolverErrchk(((FuncSolv)function)(handle, i_size, nnz, descr, vals,
                                        rowPtr, colPtr, b, 0.0, 0, xOut,
                                        singularOut));
    // TODO : SymOptimization
}

__host__ D_SparseMatrix::~D_SparseMatrix() {
    if (nnz > 0)
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
