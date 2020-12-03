#include <dataStructures/hd_data.hpp>
#include <dataStructures/matrix_element.hpp>
#include <dataStructures/sparse_matrix.hpp>

__host__ __device__ matrix_elm::matrix_elm(int k, const d_spmatrix *matrix)
    : k(k), matrix(matrix), val(matrix->data + k) {
    updateIandJ();
}
__host__ __device__ matrix_elm::matrix_elm(const d_spmatrix *matrix)
    : matrix_elm(0, matrix) {}

__host__ __device__ bool matrix_elm::has_next() {
    return k < this->matrix->nnz;
}

__host__ __device__ void matrix_elm::next() { jump(1); }
__host__ __device__ void matrix_elm::jump(int hop) {
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

__host__ __device__ void matrix_elm::print() const {
    printf("%i, %i: %f\n", i, j, *val);
}

__host__ std::string matrix_elm::to_string() const {
    char buffer[50];
    T *valHost = new T[1];
    cudaMemcpy(valHost, val, sizeof(T),
               (matrix->is_device) ? cudaMemcpyDeviceToHost
                                   : cudaMemcpyHostToHost);
    sprintf(buffer, "%i, %i: %f\n", i, j, valHost[0]);
    std::string ret_string = std::string(buffer);
    delete[] valHost;
    return ret_string;
}

__global__ void updateIandJK(const d_spmatrix *matrix, int *i, int *j, int k) {
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

__host__ __device__ void matrix_elm::updateIandJ() {
#ifndef __CUDA_ARCH__
    if (matrix->is_device) {
        hd_data<int> d_i(i);
        hd_data<int> d_j(j);
        updateIandJK<<<1, 1>>>(matrix->_device, &d_i(true), &d_j(true), k);
        cudaDeviceSynchronize();
        d_i.update_host();
        d_j.update_host();
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
