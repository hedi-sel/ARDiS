#include <assert.h>
#include <sstream>

#include "dataStructures/helper/matrix_helper.h"
#include "dataStructures/matrix_element.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "hd_data.hpp"
#include "helper/cuda/cuda_error_check.h"
#include "helper/cuda/cuda_reduction_operation.hpp"
#include "helper/cuda/cuda_thread_manager.hpp"
#include "helper/cuda/cusolverSP_error_check.h"
#include "helper/cuda/cusparse_error_check.h"
#include "matrixOperations/basic_operations.hpp"
#include "matrixOperations/row_ordering.hpp"

__host__ D_SparseMatrix::D_SparseMatrix() : D_SparseMatrix(0, 0){};

__host__ D_SparseMatrix::D_SparseMatrix(int rows, int cols, int nnz,
                                        MatrixType type, bool isDevice)
    : nnz(nnz), rows(rows), cols(cols), isDevice(isDevice), type(type),
      loaded_elements(nnz) {
    MemAlloc();
}

__host__ D_SparseMatrix::D_SparseMatrix(const D_SparseMatrix &m,
                                        bool copyToOtherMem)
    : D_SparseMatrix(m.rows, m.cols, m.nnz, m.type,
                     m.isDevice ^ copyToOtherMem) {
    loaded_elements = m.loaded_elements;
    assert(m.loaded_elements == m.nnz);
    cudaMemcpyKind memCpy =
        (m.isDevice)
            ? (isDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost
            : (isDevice) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    gpuErrchk(cudaMemcpy(data, m.data, sizeof(T) * nnz, memCpy));
    gpuErrchk(cudaMemcpy(colPtr, m.colPtr,
                         sizeof(int) * ((type == CSC) ? cols + 1 : nnz),
                         memCpy));
    gpuErrchk(cudaMemcpy(rowPtr, m.rowPtr,
                         sizeof(int) * ((type == CSR) ? rows + 1 : nnz),
                         memCpy));
}

__host__ void D_SparseMatrix::operator=(const D_SparseMatrix &other) {
    assert(isDevice == isDevice);
    MemFree();
    nnz = other.nnz;
    rows = other.rows;
    cols = other.cols;
    loaded_elements = other.loaded_elements;
    type = other.type;
    MemAlloc();
    cudaMemcpyKind memCpy =
        (isDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToHost;
    gpuErrchk(cudaMemcpy(data, other.data, sizeof(T) * nnz, memCpy));
    gpuErrchk(cudaMemcpy(colPtr, other.colPtr,
                         sizeof(int) * ((type == CSC) ? cols + 1 : nnz),
                         memCpy));
    gpuErrchk(cudaMemcpy(rowPtr, other.rowPtr,
                         sizeof(int) * ((type == CSR) ? rows + 1 : nnz),
                         memCpy));
}

__host__ bool D_SparseMatrix::operator==(const D_SparseMatrix &other) {
    if (isDevice) {
        HDData<bool> result(true);
        isEqual<<<1, 1>>>(*(this->_device), *(other._device), result(true));
        result.SetHost();
        return result();
        gpuErrchk(cudaDeviceSynchronize());
    } else
        return isEqualBody(*this, other);
}

__host__ void D_SparseMatrix::MemAlloc() {
    if (nnz == 0)
        return;
    int rowPtrSize = (type == CSR) ? rows + 1 : nnz;
    int colPtrSize = (type == CSC) ? cols + 1 : nnz;
    if (isDevice) {
        gpuErrchk(cudaMalloc(&data, nnz * sizeof(T)));
        gpuErrchk(cudaMalloc(&rowPtr, rowPtrSize * sizeof(int)));
        gpuErrchk(cudaMalloc(&colPtr, colPtrSize * sizeof(int)));
        gpuErrchk(cudaMalloc(&_device, sizeof(D_SparseMatrix)));
        gpuErrchk(cudaMemcpy(_device, this, sizeof(D_SparseMatrix),
                             cudaMemcpyHostToDevice));
    } else {
        data = new T[nnz];
        rowPtr = new int[rowPtrSize];
        for (int i = 0; i < rowPtrSize; i++)
            rowPtr[i] = 0;
        colPtr = new int[colPtrSize];
        for (int i = 0; i < colPtrSize; i++)
            colPtr[i] = 0;
    }
}
__host__ void D_SparseMatrix::MemFree() {
    if (nnz > 0)
        if (isDevice) {
            gpuErrchk(cudaFree(data));
            gpuErrchk(cudaFree(rowPtr));
            gpuErrchk(cudaFree(colPtr));
            gpuErrchk(cudaFree(_device));
        } else {
            delete[] data;
            delete[] rowPtr;
            delete[] colPtr;
        }
}

__host__ std::string D_SparseMatrix::ToString() {
    int printCount = 5;
    std::stringstream strs;

    char buffer[50];
    sprintf(buffer, "Matrix :\n%i %i %i/%i isDev=%i format=", rows, cols,
            loaded_elements, nnz, isDevice);
    strs << std::string(buffer);

    switch (type) {
    case COO:
        strs << "COO\n";
        break;
    case CSR:
        strs << "CSR\n";
        break;
    case CSC:
        strs << "CSC\n";
        break;
    }
    for (MatrixElement elm(this); elm.HasNext(); elm.Next()) {
        strs << elm.ToString();
        printCount--;
        if (printCount <= 0) {
            if (elm.HasNext())
                strs << "...\n";
            break;
        }
    }
    return strs.str();
}

__host__ __device__ void D_SparseMatrix::Print(int printCount) const {
#ifndef __CUDA_ARCH__
    if (isDevice) {
        printMatrix<<<1, 1>>>(_device, printCount);
        gpuErrchk(cudaDeviceSynchronize());
    } else
#endif
        printMatrixBody(this, printCount);
}

__host__ void D_SparseMatrix::SetNNZ(int nnz) {
    MemFree();
    this->nnz = nnz;
    this->loaded_elements = nnz;
    MemAlloc();
}

__host__ void D_SparseMatrix::StartFilling() {
    loaded_elements = 0;
    if (isDevice) {
        gpuErrchk(cudaFree(_device));
        gpuErrchk(cudaMalloc(&_device, sizeof(D_SparseMatrix)));
        gpuErrchk(cudaMemcpy(_device, this, sizeof(D_SparseMatrix),
                             cudaMemcpyHostToDevice));
        loaded_elements = nnz;
    }
}

__host__ __device__ void D_SparseMatrix::AddElement(int i, int j, T val) {
#ifndef __CUDA_ARCH__
    if (isDevice) {
        AddElementK<<<1, 1>>>(_device, i, j, val);
        gpuErrchk(cudaDeviceSynchronize());
    } else
#endif
        AddElementBody(this, i, j, val);
}

// Get the value at index k of the sparse matrix
__host__ __device__ const T &D_SparseMatrix::Get(int k) const {
    return data[k];
}
__host__ __device__ const T &D_SparseMatrix::GetLine(int i) const {
    if (type != CSR) {
        printf("Error! Doesn't work with other type than CSR");
    }
    return data[rowPtr[i]];
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
    int newSize = (toType == CSR) ? rows + 1 : cols + 1;
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

__host__ cusparseMatDescr_t D_SparseMatrix::MakeDescriptor() {
    cusparseMatDescr_t descr;
    cusparseErrchk(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    return descr;
}

__host__ cusparseSpMatDescr_t D_SparseMatrix::MakeSpDescriptor() {
    cusparseSpMatDescr_t descr;
    cusparseErrchk(cusparseCreateCsr(
        &descr, rows, cols, nnz, rowPtr, colPtr, data, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, T_Cuda));
    return std::move(descr);
}

__host__ bool D_SparseMatrix::IsSymetric() {
    bool *_return = new bool;
    if (isDevice) {
        bool *_returnGpu;
        gpuErrchk(cudaMalloc(&_returnGpu, sizeof(bool)));
        IsSymetricKernel<<<1, 1>>>(_device, _returnGpu);
        gpuErrchk(cudaDeviceSynchronize());
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
        printf("This function is not complete\n");
    } else {
        if (pointer1)
            if (pointer2) {
                cusparseErrchk(((FuncSpar)function)(handle, rows, cols, nnz,
                                                    rowPtr, colPtr, pointer1,
                                                    pointer2));
            } else {
                cusparseErrchk(((FuncSpar)function)(handle, rows, cols, nnz,
                                                    rowPtr, colPtr, pointer1));
            }
        else
            printf("This function is not complete\n");
    }
}

typedef cusolverStatus_t (*FuncSolv)(...);
__host__ void D_SparseMatrix::OperationCuSolver(void *function,
                                                cusolverSpHandle_t &handle,
                                                cusparseMatDescr_t descr, T *b,
                                                T *xOut, int *singularOut) {
    cusolverErrchk(((FuncSolv)function)(handle, rows, nnz, descr, data, rowPtr,
                                        colPtr, b, 0.0, 0, xOut, singularOut));
    // TODO : SymOptimization
}

__host__ void D_SparseMatrix::MakeDataWidth() {
    if (dataWidth >= 0)
        printf("Warning! Data width has already been computed.\n");
    dim3Pair threadblock = Make1DThreadBlock(rows);
    D_Vector width(rows);
    GetDataWidthK<<<threadblock.block, threadblock.thread>>>(
        *_device, *(D_Vector *)width._device);
    ReductionOperation(width, maximum);
    T dataWidthFloat;
    cudaMemcpy(&dataWidthFloat, width.data, sizeof(T), cudaMemcpyDeviceToHost);
    dataWidth = (int)dataWidthFloat;
}

__host__ D_SparseMatrix::~D_SparseMatrix() { MemFree(); }
