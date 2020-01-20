#include <sparseDataStruct/matrix_element.hpp>
#include <sparseDataStruct/matrix_sparse.hpp>

__host__ __device__ MatrixElement::MatrixElement(int k,
                                                 const MatrixSparse *matrix)
    : k(k), matrix(matrix), val(&matrix->vals[k]) {
    updateIandJ();
}
__host__ __device__ MatrixElement::MatrixElement(const MatrixSparse *matrix)
    : MatrixElement(0, matrix) {}

__host__ __device__ bool MatrixElement::HasNext() {
    return k < this->matrix->loaded_elements;
}

__host__ __device__ void MatrixElement::Next() {
    k++;
    if (k >= this->matrix->loaded_elements) {
        val = 0;
        i = matrix->i_size;
        j = matrix->j_size;
    } else {
        val = &val[1];
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
