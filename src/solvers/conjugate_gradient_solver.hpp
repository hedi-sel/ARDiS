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
    D_Array q;
    D_Array r;
    D_Array p;

    HDData<T> value;
    HDData<T> alpha;
    HDData<T> beta;
    HDData<T> diff;

    ChronoProfiler profiler;

    CGSolver(int n);
    bool CGSolve(D_SparseMatrix &d_mat, D_Array &b, D_Array &y, T epsilon,
                 std::string str = "");
    static bool StaticCGSolve(D_SparseMatrix &d_mat, D_Array &b, D_Array &y,
                              T epsilon); // TODO FactorizeCode
};