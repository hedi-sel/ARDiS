#include "pybind11_include.hpp"

#include "sparseDataStruct/matrix_sparse.hpp"

py::array_t<double> SolveLinEq(MatrixSparse &d_mat, py::array_t<double> &bVec);

__host__ MatrixSparse ReadFromFile(const std::string filepath);