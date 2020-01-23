#include <dataStructures/matrix_element.hpp>
#include <dataStructures/sparse_matrix.hpp>

__host__ __device__ MatrixElement::MatrixElement(int k,
                                                 const D_SparseMatrix *matrix)
    : k(k), matrix(matrix), val(&matrix->vals[k]) {
    updateIandJ();
}
__host__ __device__ MatrixElement::MatrixElement(const D_SparseMatrix *matrix)
    : MatrixElement(0, matrix) {}

__host__ __device__ bool MatrixElement::HasNext() {
    return k < this->matrix->loaded_elements;
}

__host__ __device__ void MatrixElement::Next() { Jump(1); }
__host__ __device__ void MatrixElement::Jump(int hop) {
    k += hop;
    if (hop != 0)
        if (k >= this->matrix->loaded_elements) {
            k = this->matrix->loaded_elements;
            val = 0;
            i = matrix->rows;
            j = matrix->cols;
        } else {
            val = &val[hop];
            updateIandJ();
        }
}

__host__ __device__ void MatrixElement::Print() const {
    printf("%i, %i: %f\n", i, j, *val);
}

__host__ __device__ void MatrixElement::updateIandJ() {
    if (matrix->type == CSR) {
        while (matrix->rowPtr[i + 1] <= k)
            i++;
    } else {
        i = matrix->rowPtr[k];
    }
    if (matrix->type == CSC) {
        while (matrix->colPtr[j + 1] <= k)
            j++;
    } else {
        j = matrix->colPtr[k];
    }
}
