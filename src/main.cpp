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

D_Array Test(D_SparseMatrix &d_stiff, D_SparseMatrix &d_damp, T tau,
             D_Array &d_u, T epsilon) {
    D_Array d_b(d_u.n, true);
    Dot(d_damp, d_u, d_b);
    D_Array d_x(d_u.n, true);

    D_SparseMatrix M(d_stiff.i_size, d_stiff.j_size, 0, COO, true);
    HDData<T> d_tau(-tau);
    MatrixSum(d_damp, d_stiff, d_tau(true), M);

    CGSolve(M, d_b, d_x, epsilon);

#ifndef NDEBUG
    D_Array d_y(d_stiff.i_size, true);
    Dot(M, d_x, d_y);
    alpha() = -1;
    alpha.SetDevice();
    VectorSum(d_y, d_b, alpha(true), d_x);
    HDData<T> norm;
    Dot(d_x, d_x, norm(true));
    norm.SetHost();

    printf("Norm of difference: %f\n", norm());
#endif
    return std::move(d_x);
}

D_Array SolveConjugateGradient(D_SparseMatrix &d_mat, D_Array &d_x, T epsilon) {
    D_Array d_y(d_mat.i_size, true);
    CGSolve(d_mat, d_x, d_y, epsilon);

#ifndef NDEBUG
    D_Array vec(d_x.n, true);
    Dot(d_mat, d_y, vec, true);
    HDData<T> m1(-1);
    VectorSum(d_x, vec, m1(true), vec, true);
    Dot(vec, vec, m1(true), true);
    m1.SetHost();
    printf("Norme de la difference: %f\n", m1());
#endif
    return std::move(d_y);
}

D_Array SolveCholesky(D_SparseMatrix &d_mat, py::array_t<double> &bVec) {
    assert(bVec.size() == d_mat.i_size);
    assert(d_mat.isDevice);

    D_Array b(bVec.size(), true);
    gpuErrchk(cudaMemcpy(b.vals, bVec.data(), sizeof(T) * b.n,
                         cudaMemcpyHostToDevice));
    D_Array x(d_mat.i_size, true);
    solveLinEqBody(d_mat, b, x);

    return std::move(x);
}
