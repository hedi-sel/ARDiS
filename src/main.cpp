#include <boost/timer/timer.hpp>
#include <cstdio>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include "cudaHelper/cusolverSP_error_check.h"
#include "cudaHelper/cusparse_error_check.h"
#include "cusparseOperations/row_ordering.hpp"
#include "cusparseOperations/solver.h"
#include "main.hpp"
#include "sparseDataStruct/matrix_sparse.hpp"
#include "sparseDataStruct/read_mtx_file.h"
#include "sparseDataStruct/vector_dense.hpp"

using boost::timer::cpu_timer;

py::array_t<double> SolveLinEq(MatrixSparse &d_mat, py::array_t<double> &bVec) {
    assert(bVec.size() == d_mat.i_size);
    assert(d_mat.isDevice);

    VectorDense b(bVec.size(), true);
    gpuErrchk(cudaMemcpy(b.vals, bVec.data(), sizeof(T) * b.n,
                         cudaMemcpyHostToDevice));

    VectorDense x(d_mat.i_size, true);

    // d_mat.Print();
    // b.Print();
    // x.Print();

    cpu_timer timer;
    solveLinEqBody(d_mat, b, x);
    double run_time = static_cast<double>(timer.elapsed().wall) * 1.0e-9;
    std::cout << " -Real GPU Runtime: " << run_time << "s\n";

    VectorDense result(x, true);
    return py::array_t(result.n, result.vals);
}
