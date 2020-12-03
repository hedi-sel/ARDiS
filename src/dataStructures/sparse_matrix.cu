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

__host__ d_spmatrix::d_spmatrix() : d_spmatrix(0, 0){};

__host__ d_spmatrix::d_spmatrix(int rows, int cols, int nnz, matrix_type type,
                                bool is_device)
    : nnz(nnz), rows(rows), cols(cols), is_device(is_device), type(type),
      loaded_elements(nnz) {
    mem_alloc();
}

__host__ d_spmatrix::d_spmatrix(const d_spmatrix &m, bool copyToOtherMem)
    : d_spmatrix(m.rows, m.cols, m.nnz, m.type, m.is_device ^ copyToOtherMem) {
    loaded_elements = m.loaded_elements;
    assert(m.loaded_elements == m.nnz);
    cudaMemcpyKind memCpy =
        (m.is_device)
            ? (is_device) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost
            : (is_device) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    gpuErrchk(cudaMemcpy(data, m.data, sizeof(T) * nnz, memCpy));
    gpuErrchk(cudaMemcpy(colPtr, m.colPtr,
                         sizeof(int) * ((type == CSC) ? cols + 1 : nnz),
                         memCpy));
    gpuErrchk(cudaMemcpy(rowPtr, m.rowPtr,
                         sizeof(int) * ((type == CSR) ? rows + 1 : nnz),
                         memCpy));
}

__host__ void d_spmatrix::operator=(const d_spmatrix &other) {
    assert(is_device == is_device);
    mem_free();
    nnz = other.nnz;
    rows = other.rows;
    cols = other.cols;
    loaded_elements = other.loaded_elements;
    type = other.type;
    mem_alloc();
    cudaMemcpyKind memCpy =
        (is_device) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToHost;
    gpuErrchk(cudaMemcpy(data, other.data, sizeof(T) * nnz, memCpy));
    gpuErrchk(cudaMemcpy(colPtr, other.colPtr,
                         sizeof(int) * ((type == CSC) ? cols + 1 : nnz),
                         memCpy));
    gpuErrchk(cudaMemcpy(rowPtr, other.rowPtr,
                         sizeof(int) * ((type == CSR) ? rows + 1 : nnz),
                         memCpy));
}

__host__ bool d_spmatrix::operator==(const d_spmatrix &other) {
    if (is_device) {
        hd_data<bool> result(true);
        is_equalK<<<1, 1>>>(*(this->_device), *(other._device), result(true));
        result.update_host();
        return result();
        gpuErrchk(cudaDeviceSynchronize());
    } else
        return is_equalBody(*this, other);
}

__host__ void d_spmatrix::mem_alloc() {
    if (nnz == 0)
        return;
    int rowPtrSize = (type == CSR) ? rows + 1 : nnz;
    int colPtrSize = (type == CSC) ? cols + 1 : nnz;
    if (is_device) {
        gpuErrchk(cudaMalloc(&data, nnz * sizeof(T)));
        gpuErrchk(cudaMalloc(&rowPtr, rowPtrSize * sizeof(int)));
        gpuErrchk(cudaMalloc(&colPtr, colPtrSize * sizeof(int)));
        gpuErrchk(cudaMalloc(&_device, sizeof(d_spmatrix)));
        gpuErrchk(cudaMemcpy(_device, this, sizeof(d_spmatrix),
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
__host__ void d_spmatrix::mem_free() {
    if (nnz > 0)
        if (is_device) {
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

__host__ std::string d_spmatrix::to_string() {
    int printCount = 5;
    std::stringstream strs;

    char buffer[50];
    sprintf(buffer, "Matrix :\n%i %i %i/%i isDev=%i format=", rows, cols,
            loaded_elements, nnz, is_device);
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
    for (matrix_elm elm(this); elm.has_next(); elm.next()) {
        strs << elm.to_string();
        printCount--;
        if (printCount <= 0) {
            if (elm.has_next())
                strs << "...\n";
            break;
        }
    }
    return strs.str();
}

__host__ __device__ void d_spmatrix::print(int printCount) const {
#ifndef __CUDA_ARCH__
    if (is_device) {
        print_matrixK<<<1, 1>>>(_device, printCount);
        gpuErrchk(cudaDeviceSynchronize());
    } else
#endif
        print_matrixBody(this, printCount);
}

__host__ void d_spmatrix::set_nnz(int nnz) {
    mem_free();
    this->nnz = nnz;
    this->loaded_elements = nnz;
    mem_alloc();
}

__host__ void d_spmatrix::start_filling() {
    loaded_elements = 0;
    if (is_device) {
        gpuErrchk(cudaFree(_device));
        gpuErrchk(cudaMalloc(&_device, sizeof(d_spmatrix)));
        gpuErrchk(cudaMemcpy(_device, this, sizeof(d_spmatrix),
                             cudaMemcpyHostToDevice));
        loaded_elements = nnz;
    }
}

__host__ __device__ void d_spmatrix::add_element(int i, int j, T val) {
#ifndef __CUDA_ARCH__
    if (is_device) {
        add_elementK<<<1, 1>>>(_device, i, j, val);
        gpuErrchk(cudaDeviceSynchronize());
    } else
#endif
        add_elementBody(this, i, j, val);
}

// Get the value at index k of the sparse matrix
__host__ __device__ const T &d_spmatrix::get(int k) const { return data[k]; }
__host__ __device__ const T &d_spmatrix::get_line(int i) const {
    if (type != CSR) {
        printf("Error! Doesn't work with other type than CSR");
    }
    return data[rowPtr[i]];
}

__host__ __device__ T d_spmatrix::lookup(int i, int j) const {
    for (matrix_elm elm(this); elm.has_next(); elm.next())
        if (elm.i == i && elm.j == j)
            return *elm.val;
    return 0;
}

__host__ void d_spmatrix::to_compress_dtype(matrix_type toType) {
    if (toType == COO) {
        if (is_convertible_to(CSR))
            toType = CSR;
        else if (is_convertible_to(CSC))
            toType = CSC;
        else {
            printf("Not convertible to any type!\n");
            return;
        }
    } else {
        assert(is_convertible_to(toType));
    }
    int newSize = (toType == CSR) ? rows + 1 : cols + 1;
    int *newArray;
    if (is_device) {
        gpuErrchk(cudaMalloc(&newArray, newSize * sizeof(int)));
        convert_arrayK<<<1, 1>>>(_device, (toType == CSR) ? rowPtr : colPtr,
                                 newArray, newSize);
        cudaFree((toType == CSR) ? rowPtr : colPtr);
    } else {
        newArray = new int[newSize];
        convert_arrayBody(this, (toType == CSR) ? rowPtr : colPtr, newArray,
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
    if (is_device) {
        loaded_elements = nnz; // Warning!! There is no assert to protect this!
        gpuErrchk(cudaMemcpy(_device, this, sizeof(d_spmatrix),
                             cudaMemcpyHostToDevice));
        gpuErrchk(cudaDeviceSynchronize());
    }
}

__host__ bool d_spmatrix::is_convertible_to(matrix_type toType) const {
    assert(toType != type);
    if (toType == COO)
        return true;
    if (type != COO)
        return false;
    int *analyzedArray = (toType == CSR) ? rowPtr : colPtr;
    bool isOK = true;
    if (is_device) {
        bool *_isOK;
        gpuErrchk(cudaMalloc(&_isOK, sizeof(bool)));
        checkOrdered<<<1, 1>>>(analyzedArray, nnz, _isOK);
        gpuErrchk(
            cudaMemcpy(&isOK, _isOK, sizeof(bool), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(_isOK));
        gpuErrchk(cudaDeviceSynchronize());
    } else {
        check_orderedBody(analyzedArray, nnz, &isOK);
    }
    return isOK;
}

__host__ void d_spmatrix::to_csr() {
    if (type == CSR)
        throw("Error! Already CSR type \n");
    if (type == CSC)
        throw("Error! Already CSC type \n");
    if (!is_convertible_to(CSR)) {
        RowOrdering(*this);
    }
    assert(is_convertible_to(CSR));
    to_compress_dtype(CSR);
    assert(type == CSR);
}

__host__ cusparseMatDescr_t d_spmatrix::make_descriptor() {
    cusparseMatDescr_t descr;
    cusparseErrchk(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    return descr;
}

__host__ cusparseSpMatDescr_t d_spmatrix::make_sp_descriptor() {
    cusparseSpMatDescr_t descr;
    cusparseErrchk(cusparseCreateCsr(
        &descr, rows, cols, nnz, rowPtr, colPtr, data, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, T_Cuda));
    return std::move(descr);
}

__host__ bool d_spmatrix::is_symetric() {
    bool *_return = new bool;
    if (is_device) {
        bool *_returnGpu;
        gpuErrchk(cudaMalloc(&_returnGpu, sizeof(bool)));
        is_symetricK<<<1, 1>>>(_device, _returnGpu);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(_return, _returnGpu, sizeof(bool),
                             cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(_returnGpu));
        gpuErrchk(cudaDeviceSynchronize());
    } else {
        is_symetricBody(this, _return);
    }
    return *_return;
}

typedef cusparseStatus_t (*FuncSpar)(...);
__host__ void d_spmatrix::operation_cusparse(void *function,
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
__host__ void d_spmatrix::operation_cusolver(void *function,
                                             cusolverSpHandle_t &handle,
                                             cusparseMatDescr_t descr, T *b,
                                             T *xOut, int *singularOut) {
    cusolverErrchk(((FuncSolv)function)(handle, rows, nnz, descr, data, rowPtr,
                                        colPtr, b, 0.0, 0, xOut, singularOut));
    // TODO : SymOptimization
}

__host__ void d_spmatrix::make_datawidth() {
    if (dataWidth >= 0)
        printf("Warning! Data width has already been computed.\n");
    dim3Pair threadblock = make1DThreadBlock(rows);
    d_vector width(rows);
    get_datawidthK<<<threadblock.block, threadblock.thread>>>(
        *_device, *(d_vector *)width._device);
    ReductionOperation(width, maximum);
    T dataWidthFloat;
    cudaMemcpy(&dataWidthFloat, width.data, sizeof(T), cudaMemcpyDeviceToHost);
    dataWidth = (int)dataWidthFloat;
}

__host__ d_spmatrix::~d_spmatrix() { mem_free(); }
