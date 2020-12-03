#include "cuda_runtime.h"
#include "dataStructures/array.hpp"
#include "dataStructures/matrix_element.hpp"
#include "dataStructures/sparse_matrix.hpp"

__device__ __host__ void convert_arrayBody(d_spmatrix *matrix,
                                           int *toOrderArray, int *newArray,
                                           int newSize) {
    newArray[0] = 0;
    int it = 0;
    for (int k = 1; k < newSize; k++) {
        newArray[k] = newArray[k - 1];
        while (it < matrix->nnz && toOrderArray[it] == k - 1) {
            newArray[k]++;
            it++;
        }
    }
}

__global__ void convert_arrayK(d_spmatrix *matrix, int *toOrderArray,
                               int *newArray, int newSize) {
    convert_arrayBody(matrix, toOrderArray, newArray, newSize);
}

__device__ __host__ void check_orderedBody(int *array, int size, bool *_isOK) {
    bool isOK = true;
    for (int k = 1; k < size && isOK; k++) {
        isOK = isOK && array[k] >= array[k - 1];
    }
    *_isOK = isOK;
};

__global__ void checkOrdered(int *array, int size, bool *_isOK) {
    check_orderedBody(array, size, _isOK);
}

__device__ __host__ inline void print_matrixBody(const d_spmatrix *matrix,
                                                 int printCount = 0) {
    printf("Matrix :\n%i %i %i/%i isDev=%i format=", matrix->rows, matrix->cols,
           matrix->loaded_elements, matrix->nnz, matrix->is_device);
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
    for (matrix_elm elm(matrix); elm.has_next(); elm.next()) {
        elm.print();
        if (printCount > 0 && !(elm.k < printCount - 1)) {
            printf("... \n");
            return;
        }
    }
}

__global__ void print_matrixK(const d_spmatrix *matrix, int printCount) {
    print_matrixBody(matrix, printCount);
}

__device__ __host__ void is_symetricBody(const d_spmatrix *matrix,
                                         bool *_return) {
    *_return = true;
    for (matrix_elm elm(matrix); elm.has_next(); elm.next()) {
        if (elm.i != elm.j && matrix->lookup(elm.j, elm.i) != *elm.val) {
            *_return = false;
            return;
        }
    }
    return;
}

__global__ void is_symetricK(const d_spmatrix *matrix, bool *_return) {
    is_symetricBody(matrix, _return);
}

__device__ __host__ void add_elementBody(d_spmatrix *m, int i, int j, T &val) {
    if (m->loaded_elements >= m->nnz) {
        printf("Error! The Sparse Matrix exceeded its memory allocation! At:"
               " i=%i j=%i val=%f\n",
               i, j, val);
        return;
    }
    m->data[m->loaded_elements] = val;
    if (m->type == CSC) {
        if (m->colPtr[j + 1] == 0)
            m->colPtr[j + 1] = m->colPtr[j];
        m->colPtr[j + 1]++;
    } else {
        m->colPtr[m->loaded_elements] = j;
    }
    if (m->type == CSR) {
        if (m->rowPtr[i + 1] == 0)
            m->rowPtr[i + 1] = m->rowPtr[i];
        m->rowPtr[i + 1]++;
    } else {
        m->rowPtr[m->loaded_elements] = i;
    }
    m->loaded_elements++;
}

__global__ void add_elementK(d_spmatrix *m, int i, int j, T &val) {
    add_elementBody(m, i, j, val);
}

__global__ void get_datawidthK(d_spmatrix &d_mat, d_vector &width) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= d_mat.rows)
        return;
    matrix_elm it(d_mat.rowPtr[i], &d_mat);
    width.data[i] = 0;
    do {
        width.data[i] += 1;
        it.next();
    } while (it.i == i && it.has_next());
}

__device__ __host__ bool is_equalBody(const d_spmatrix &m1,
                                      const d_spmatrix &m2) {
    if (m1.nnz != m2.nnz || m1.cols != m2.cols || m1.rows != m2.rows ||
        m1.type != m2.type || m1.is_device != m2.is_device) {
        printf("debut");
        return false;
    }

    for (int i = 0; i < m1.nnz; i++)
        if (m1.data[i] != m2.data[i]) {
            printf("rows %i: %e != %e, %e == %e\nimax = %i\n", i, m1.data[i],
                   m2.data[i], m1.data[i - 1], m2.data[i - 1], m1.nnz);
            return false;
        }

    for (int i = 0; i < ((m1.type == CSR) ? m1.rows + 1 : m1.nnz); i++)
        if (m1.rowPtr[i] != m2.rowPtr[i]) {
            printf("rows %i: %i %i, %i == %i\nimax = %i\n", i, m1.rowPtr[i],
                   m2.rowPtr[i], m1.rowPtr[i - 1], m2.rowPtr[i - 1],
                   (m1.type == CSR) ? m1.rows + 1 : m1.nnz);
            return false;
        }
    for (int i = 0; i < ((m1.type == CSC) ? m1.cols + 1 : m1.nnz); i++)
        if (m1.colPtr[i] != m2.colPtr[i]) {
            printf("rows %i: %i %i, %i == %i\nimax = %i\n", i, m1.colPtr[i],
                   m2.colPtr[i], m1.colPtr[i - 1], m2.colPtr[i - 1],
                   (m1.type == CSR) ? m1.cols + 1 : m1.nnz);
            return false;
        }
    return true;
}

__global__ void is_equalK(const d_spmatrix &m1, const d_spmatrix &m2,
                          bool &result) {
    result = is_equalBody(m1, m2);
}