#pragma once

#include "cuda_runtime.h"

#include "constants.hpp"
#include "dataStructures/array.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "matrixOperations/basic_operations.hpp"

class CGSolver {
  public:
    int n;
    D_Vector q;
    D_Vector r;
    D_Vector p;

    HDData<T> value;
    HDData<T> alpha;
    HDData<T> beta;
    HDData<T> diff;

#ifndef NDEBUG_PROFILING
    ChronoProfiler profiler;
#endif

    CGSolver(int n);
    bool CGSolve(D_SparseMatrix &d_mat, D_Vector &b, D_Vector &y, T epsilon,
                 std::string str = "");
    int n_iter_last = 0;
    static bool StaticCGSolve(D_SparseMatrix &d_mat, D_Vector &b, D_Vector &y,
                              T epsilon); // TODO FactorizeCode
};