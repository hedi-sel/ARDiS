#pragma once

#include <cstdarg>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <cusparse_v2.h>
#include <string>
#include <utility>

#include "constants.hpp"

enum matrix_type { COO, CSR, CSC };

class d_spmatrix {
  public:
    int nnz;
    int rows;
    int cols;

    int loaded_elements = 0;
    int dataWidth = -1;

    matrix_type type;
    const bool is_device;
    // cusparseMatDescr_t descr = NULL;

    T *data;
    int *rowPtr;
    int *colPtr;
    d_spmatrix *_device;

    __host__ d_spmatrix();
    __host__ d_spmatrix(int rows, int cols, int nnz = 0, matrix_type = COO,
                        bool is_device = true);
    __host__ d_spmatrix(const d_spmatrix &, bool copyToOtherMem = false);
    __host__ void operator=(const d_spmatrix &);
    __host__ bool operator==(const d_spmatrix &);

    __host__ ~d_spmatrix();

    __host__ std::string to_string();
    __host__ __device__ void print(int printCount = 5) const;

    __host__ void set_nnz(int);

    // Add an element at index k
    __host__ void start_filling();
    __host__ __device__ void add_element(int i, int j, T);

    // __host__ __device__ matrix_elm start() const;

    // Get the value at index k of the sparse matrix
    __host__ __device__ const T &get(int k) const;
    __host__ __device__ const T &get_line(int i) const;

    // Get the element at position (i, j) in the matrix, if it is defined
    __host__ __device__ T lookup(int i, int j) const;

    // Turn the matrix to CSC or CSR type
    __host__ void to_compress_dtype(matrix_type = COO);
    __host__ bool is_convertible_to(matrix_type) const;

    __host__ void to_csr();

    __host__ bool is_symetric();

    __host__ cusparseMatDescr_t make_descriptor();
    __host__ cusparseSpMatDescr_t make_sp_descriptor();

    __host__ void operation_cusparse(void *function, cusparseHandle_t &,
                                     bool addValues = false, void * = NULL,
                                     void * = NULL);
    __host__ void operation_cusolver(void *function, cusolverSpHandle_t &,
                                     cusparseMatDescr_t, T *b = NULL,
                                     T *xOut = NULL, int *singularOut = NULL);

    __host__ void make_datawidth();

  private:
    __host__ void mem_alloc();
    __host__ void mem_free();
};