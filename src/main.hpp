#include "pybind11_include.hpp"

#include "sparseDataStruct/matrix_sparse.hpp"

D_Array SolveCholesky(D_SparseMatrix &d_mat, py::array_t<double> &bVec);
D_Array SolveConjugateGradient(D_SparseMatrix &d_mat, D_Array &d_b,
                               T epsilon = 1.0e-3);
D_Array Test(D_SparseMatrix &, D_SparseMatrix &, T, D_Array &x,
             T epsilon = 1.0e-3);

__host__ D_SparseMatrix ReadFromFile(const std::string filepath);