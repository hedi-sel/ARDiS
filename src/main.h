#pragma once

#include <boost/timer/timer.hpp>
#include <cstdio>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include "dataStructures/array.hpp"
#include "dataStructures/read_mtx_file.h"
#include "dataStructures/sparse_matrix.hpp"
#include "hediHelper/cuda/cusolverSP_error_check.h"
#include "hediHelper/cuda/cusparse_error_check.h"
#include "matrixOperations/basic_operations.hpp"
#include "matrixOperations/row_ordering.hpp"
#include "solvers/conjugate_gradient_solver.hpp"
#include "solvers/inversion_solver.h"

using boost::timer::cpu_timer;

void checkSolve(D_SparseMatrix &M, D_Array &d_b, D_Array &d_x) {
    D_Array vec(d_b.n, true);
    Dot(M, d_x, vec, true);
    HDData<T> m1(-1);
    VectorSum(d_b, vec, m1(true), vec, true);
    Dot(vec, vec, m1(true), true);
    m1.SetHost();
    printf("Norme de la difference: %f\n", m1());
}

D_Array SolveConjugateGradient(D_SparseMatrix &d_mat, D_Array &d_x, T epsilon) {
    D_SparseMatrix mat(d_mat);
    D_Array d_y(d_x.n);
    CGSolve(mat, d_x, d_y, epsilon);

#ifndef NDEBUG
    checkSolve(mat, d_x, d_y);
#endif

    return d_y;
}

D_Array DiffusionTest(D_SparseMatrix &d_stiff, D_SparseMatrix &d_damp, T tau,
                      D_Array &d_u, T epsilon) {
    D_Array d_b(d_u.n, true);
    Dot(d_damp, d_u, d_b);
    D_SparseMatrix M(d_stiff.rows, d_stiff.cols, 0, COO, true);
    HDData<T> m(-tau);
    MatrixSum(d_damp, d_stiff, m(true), M);

    auto d_x = SolveConjugateGradient(M, d_b, epsilon);
    PrintDotProfiler();

    return d_x;
}

D_Array SolveCholesky(D_SparseMatrix &d_mat, py::array_t<double> &bVec) {
    assert(bVec.size() == d_mat.rows);
    assert(d_mat.isDevice);

    D_Array b(bVec.size(), true);
    gpuErrchk(cudaMemcpy(b.vals, bVec.data(), sizeof(T) * b.n,
                         cudaMemcpyHostToDevice));
    D_Array x(d_mat.rows, true);
    solveLinEqBody(d_mat, b, x);

    return std::move(x);
}
