#include "pybind11_include.hpp"

#include "sparseDataStruct/matrix_sparse.hpp"

py::array_t<double> SolveCholesky(MatrixSparse &d_mat,
                                  py::array_t<double> &bVec);
py::array_t<double> SolveConjugateGradient(MatrixSparse &d_mat,
                                           py::array_t<double> &x,
                                           bool printError = false);
py::array_t<double> Test(MatrixSparse &, MatrixSparse &,
                         py::array_t<double> &x);

__host__ MatrixSparse ReadFromFile(const std::string filepath);