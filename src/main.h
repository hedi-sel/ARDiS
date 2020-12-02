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
#include "reactionDiffusionSystem/simulation.hpp"
#include "solvers/conjugate_gradient_solver.hpp"
#include "solvers/inversion_solver.h"

void ToCSV(d_vector &array, std::string outputPath, std::string prefix = "",
           std::string suffix = "") {
    if (array.isDevice) { // If device memory, copy to host, and restart the
                          // function
        d_vector h_copy(array, true);
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
void ToCSV(state &state, std::string outputPath) {
    std::ofstream fout;
    fout.open(outputPath, std::ios_base::app);
    fout << state.vector_size << "\n";
    fout.close();
    for (auto sp : state.names) {
        ToCSV(state.vector_holder.at(sp.second), outputPath, sp.first, "\n");
    }
}

void checkSolve(d_spmatrix &M, d_vector &d_b, d_vector &d_x) {
    d_vector vec(d_b.n, true);
    dot(M, d_x, vec, true);
    hd_data<T> m1(-1);
    vector_sum(d_b, vec, m1(true), vec, true);
    dot(vec, vec, m1(true), true);
    m1.SetHost();
    printf("Norme de la difference: %f\n", m1());
}

void SolveConjugateGradient(d_spmatrix &d_mat, d_vector &d_b, T epsilon,
                            d_vector &d_x) {
    cg_solver::StaticCGSolve(d_mat, d_b, d_x, epsilon);
#ifndef NDEBUG
    checkSolve(d_mat, d_x, d_b);
#endif
    // PrintDotProfiler();
}

void DiffusionTest(d_spmatrix &d_stiff, d_spmatrix &d_damp, T tau,
                   d_vector &d_u, T epsilon) {
    d_vector d_b(d_u.n, true);
    dot(d_damp, d_u, d_b);
    d_spmatrix M(d_stiff.rows, d_stiff.cols, 0, COO, true);
    hd_data<T> m(-tau);
    matrix_sum(d_damp, d_stiff, m(true), M);
    cg_solver::StaticCGSolve(M, d_b, d_u, epsilon);
}

d_vector solve_cholesky(d_spmatrix &d_mat, py::array_t<T> &bVec) {
    assert(bVec.size() == d_mat.rows);
    assert(d_mat.isDevice);

    d_vector b(bVec.size(), true);
    gpuErrchk(cudaMemcpy(b.data, bVec.data(), sizeof(T) * b.n,
                         cudaMemcpyHostToDevice));
    d_vector x(d_mat.rows, true);
    solveLinEqBody(d_mat, b, x);

    return std::move(x);
}
