#include <dataStructures/hd_data.hpp>
#include <dataStructures/matrix_element.hpp>
#include <dataStructures/sparse_matrix.hpp>

__host__ __device__ MatrixElement::MatrixElement(int k,
                                                 const d_spmatrix *matrix)
    : k(k), matrix(matrix), val(matrix->data + k) {
    updateIandJ();
}
__host__ __device__ MatrixElement::MatrixElement(const d_spmatrix *matrix)
    : MatrixElement(0, matrix) {}

__host__ __device__ bool MatrixElement::HasNext() {
    return k < this->matrix->nnz;
}

__host__ __device__ void MatrixElement::Next() { Jump(1); }
__host__ __device__ void MatrixElement::Jump(int hop) {
    k += hop;
    if (hop != 0)
        if (k >= this->matrix->nnz) {
            k = this->matrix->nnz;
            val = 0;
            i = matrix->rows;
            j = matrix->cols;
        } else {
            val = val + hop;
            updateIandJ();
        }
}

__host__ __device__ void MatrixElement::print() const {
    printf("%i, %i: %f\n", i, j, *val);
}

__host__ std::string MatrixElement::ToString() const {
    char buffer[50];
    T *valHost = new T[1];
    cudaMemcpy(valHost, val, sizeof(T),
               (matrix->isDevice) ? cudaMemcpyDeviceToHost
                                  : cudaMemcpyHostToHost);
    sprintf(buffer, "%i, %i: %f\n", i, j, valHost[0]);
    std::string ret_string = std::string(buffer);
    delete[] valHost;
    return ret_string;
}

__global__ void launchUpdateIandJ(const d_spmatrix *matrix, int *i, int *j,
                                  int k) {
    if (matrix->type == CSR) {
        while (matrix->rowPtr[i[0] + 1] <= k)
            i[0]++;
    } else {
        i[0] = matrix->rowPtr[k];
    }

    if (matrix->type == CSC) {
        while (matrix->colPtr[j[0] + 1] <= k)
            j[0]++;
    } else {
        j[0] = matrix->colPtr[k];
    }
}

__host__ __device__ void MatrixElement::updateIandJ() {
#ifndef __CUDA_ARCH__
    if (matrix->isDevice) {
        hd_data<int> d_i(i);
        hd_data<int> d_j(j);
        launchUpdateIandJ<<<1, 1>>>(matrix->_device, &d_i(true), &d_j(true), k);
        cudaDeviceSynchronize();
        d_i.SetHost();
        d_j.SetHost();
        i = d_i();
        j = d_j();
        return;
    }
#endif
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
