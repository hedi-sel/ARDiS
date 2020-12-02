#pragma once

#include "cuda_runtime.h"

#include "constants.hpp"
#include "dataStructures/array.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "matrixOperations/basic_operations.hpp"

class cg_solver {
  public:
    int n;
    d_vector q;
    d_vector r;
    d_vector p;

    hd_data<T> value;
    hd_data<T> alpha;
    hd_data<T> beta;
    hd_data<T> diff;

#ifndef NDEBUG_PROFILING
    ChronoProfiler profiler;
#endif

    cg_solver(int n);
    bool CGSolve(d_spmatrix &d_mat, d_vector &b, d_vector &y, T epsilon,
                 std::string str = "");
    int n_iter_last = 0;
    static bool StaticCGSolve(d_spmatrix &d_mat, d_vector &b, d_vector &y,
                              T epsilon); // TODO FactorizeCode
};