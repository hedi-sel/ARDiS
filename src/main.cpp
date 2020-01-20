#include <boost/timer/timer.hpp>
#include <cstdio>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include "cudaHelper/cusolverSP_error_check.h"
#include "cudaHelper/cusparse_error_check.h"
#include "cusparseOperations/dot_sparse.hpp"
#include "cusparseOperations/row_ordering.hpp"
#include "main.hpp"
#include "solvers/conjugate_gradient_solver.hpp"
#include "solvers/inversion_solver.h"
#include "sparseDataStruct/matrix_sparse.hpp"
#include "sparseDataStruct/read_mtx_file.h"
#include "sparseDataStruct/vector_dense.hpp"

using boost::timer::cpu_timer;
py::array_t<double> Test(MatrixSparse &d_stiff, MatrixSparse &d_damp,
                         py::array_t<double> &x) {
    VectorDense d_x(x.size(), true);
    gpuErrchk(cudaMemcpy(d_x.vals, x.data(), sizeof(T) * d_x.n,
                         cudaMemcpyHostToDevice));
    VectorDense d_y(d_stiff.i_size, true);
    Dot(d_stiff, d_x, d_y);

    MatrixSparse *sum;
    MatrixSum(d_stiff, d_stiff, sum);

    // d_stiff.Print();
    // sum->Print();

    d_y.Print();
    VectorDense result(d_y, true);
    return py::array_t(result.n, result.vals);
}

py::array_t<double> SolveConjugateGradient(MatrixSparse &d_mat,
                                           py::array_t<double> &x,
                                           bool printError) {
    assert(x.size() == d_mat.j_size);
    VectorDense d_x(x.size(), true);
    gpuErrchk(cudaMemcpy(d_x.vals, x.data(), sizeof(T) * d_x.n,
                         cudaMemcpyHostToDevice));
    VectorDense d_y(d_mat.i_size, true);
    CGSolve(d_mat, d_x, d_y, 0.01);
    cudaDeviceSynchronize();
    VectorDense result(d_y, true);

    if (printError) {
        VectorDense vec(d_x.n, true);
        Dot(d_mat, d_y, vec, true);
        HDData<T> m1(-1);
        VectorSum(d_x, vec, m1(true), vec, true);
        Dot(vec, vec, m1(true), true);
        m1.SetHost();
        printf("Norme de la difference: %f\n", m1());
    }
    return py::array_t(result.n, result.vals);
}

py::array_t<double> SolveCholesky(MatrixSparse &d_mat,
                                  py::array_t<double> &bVec) {
    assert(bVec.size() == d_mat.i_size);
    assert(d_mat.isDevice);

    VectorDense b(bVec.size(), true);
    gpuErrchk(cudaMemcpy(b.vals, bVec.data(), sizeof(T) * b.n,
                         cudaMemcpyHostToDevice));
    VectorDense x(d_mat.i_size, true);

    solveLinEqBody(d_mat, b, x);

    VectorDense result(x, true);
    return py::array_t(result.n, result.vals);
}
