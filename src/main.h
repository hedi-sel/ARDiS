#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <math.h>
#include <stdio.h>

#include "dataStructures/array.hpp"
#include "dataStructures/read_mtx_file.h"
#include "dataStructures/sparse_matrix.hpp"
#include "helper/cuda/cusolverSP_error_check.h"
#include "helper/cuda/cusparse_error_check.h"
#include "matrixOperations/basic_operations.hpp"
#include "matrixOperations/row_ordering.hpp"
#include "reactionDiffusionSystem/system.hpp"
#include "solvers/conjugate_gradient_solver.hpp"
#include "solvers/inversion_solver.h"

void ToCSV(D_Vector &array, std::string outputPath, std::string prefix = "",
           std::string suffix = "") {
    if (array.isDevice) { // If device memory, copy to host, and restart the
                          // function
        D_Vector h_copy(array, true);
        ToCSV(h_copy, outputPath, prefix, suffix);
        return;
    }
    // once we made sure the vector's data is on the host memory
    std::ofstream fout;
    fout.open(outputPath, std::ios_base::app);
    if (prefix != std::string("")) {
        fout << prefix << "\t";
    }
    for (size_t j = 0; j < array.n; j++)
        fout << ((j == 0) ? "" : "\t") << array.data[j];
    if (suffix != std::string(""))
        fout << suffix;
    fout.close();
}
void ToCSV(State &state, std::string outputPath) {
    std::ofstream fout;
    fout.open(outputPath, std::ios_base::app);
    fout << state.vector_size << "\n";
    fout.close();
    for (auto sp : state.names) {
        ToCSV(state.vector_holder.at(sp.second), outputPath, sp.first, "\n");
    }
}

void checkSolve(D_SparseMatrix &M, D_Vector &d_b, D_Vector &d_x) {
    D_Vector vec(d_b.n, true);
    Dot(M, d_x, vec, true);
    HDData<T> m1(-1);
    VectorSum(d_b, vec, m1(true), vec, true);
    Dot(vec, vec, m1(true), true);
    m1.SetHost();
    printf("Norme de la difference: %f\n", m1());
}

void SolveConjugateGradient(D_SparseMatrix &d_mat, D_Vector &d_b, T epsilon,
                            D_Vector &d_x) {
    CGSolver::StaticCGSolve(d_mat, d_b, d_x, epsilon);
#ifndef NDEBUG
    checkSolve(d_mat, d_x, d_b);
#endif
    // PrintDotProfiler();
}

void DiffusionTest(D_SparseMatrix &d_stiff, D_SparseMatrix &d_damp, T tau,
                   D_Vector &d_u, T epsilon) {
    D_Vector d_b(d_u.n, true);
    Dot(d_damp, d_u, d_b);
    D_SparseMatrix M(d_stiff.rows, d_stiff.cols, 0, COO, true);
    HDData<T> m(-tau);
    MatrixSum(d_damp, d_stiff, m(true), M);
    CGSolver::StaticCGSolve(M, d_b, d_u, epsilon);
}

D_Vector SolveCholesky(D_SparseMatrix &d_mat, py::array_t<T> &bVec) {
    assert(bVec.size() == d_mat.rows);
    assert(d_mat.isDevice);

    D_Vector b(bVec.size(), true);
    gpuErrchk(cudaMemcpy(b.data, bVec.data(), sizeof(T) * b.n,
                         cudaMemcpyHostToDevice));
    D_Vector x(d_mat.rows, true);
    solveLinEqBody(d_mat, b, x);

    return std::move(x);
}
