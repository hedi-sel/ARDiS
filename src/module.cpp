#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <assert.h>

#include "cusparseOperations/dot_sparse.hpp"
#include "main.hpp"
#include "sparseDataStruct/double_data.hpp"
#include "sparseDataStruct/matrix_sparse.hpp"
#include "sparseDataStruct/vector_dense.hpp"

PYBIND11_MODULE(dna, m) {
    py::enum_<MatrixType>(m, "MatrixType")
        .value("COO", COO)
        .value("CSR", CSR)
        .value("CSC", CSC)
        .export_values();

    py::class_<D_SparseMatrix>(m, "D_SparseMatrix")
        .def(py::init<int, int, int, MatrixType>())
        .def(py::init<int, int, int>())
        .def(py::init<int, int>())
        .def(py::init<>())
        .def(py::init<const D_SparseMatrix &, bool>())
        .def("AddElement", &D_SparseMatrix::AddElement)
        .def("ConvertMatrixToCSR", &D_SparseMatrix::ConvertMatrixToCSR)
        .def("Print", &D_SparseMatrix::Print, py::arg("printCount") = 5)
        .def("Dot",
             [](D_SparseMatrix &mat, D_Array &x) {
                 D_Array y(mat.i_size);
                 Dot(mat, x, y);
                 return std::move(y);
             },
             py::return_value_policy::move)
        .def_readonly("nnz", &D_SparseMatrix::nnz)
        .def_readonly("i_size", &D_SparseMatrix::i_size)
        .def_readonly("j_size", &D_SparseMatrix::j_size)
        .def_readonly("type", &D_SparseMatrix::type)
        .def_readonly("isDevice", &D_SparseMatrix::isDevice);

    py::class_<D_Array>(m, "D_Array")
        .def(py::init<int>())
        .def(py::init<const D_Array &>())
        .def("Print", &D_Array::Print)
        .def("Fill",
             [](D_Array &v, py::array_t<T> &x) {
                 assert(x.size() == v.n);
                 gpuErrchk(cudaMemcpy(v.vals, x.data(), sizeof(T) * x.size(),
                                      cudaMemcpyHostToDevice));
             })
        .def("Dot", [](D_Array &a, D_Array &b) {
            HDData<T> res;
            Dot(a, b, res(true));
            return res();
        });

    m.doc() = "Sparse Linear Equation solving API"; // optional module docstring
    m.def("SolveCholesky", &SolveCholesky, py::return_value_policy::move);
    m.def("SolveConjugateGradient", &SolveConjugateGradient,
          py::return_value_policy::move);
    m.def("ReadFromFile", &ReadFromFile, py::return_value_policy::move);
    m.def("Test", &Test);

    m.def("GetNumpyVector",
          [](D_Array &v) {
              D_Array x(v, true);
              return std::move(py::array_t(x.n, x.vals));
          },
          py::return_value_policy::move);
    m.def("MatrixSum", [](D_SparseMatrix &a, D_SparseMatrix &b,
                          D_SparseMatrix &c) { MatrixSum(a, b, c); });
    m.def("MatrixSum",
          [](D_SparseMatrix &a, D_SparseMatrix &b, T alpha, D_SparseMatrix &c) {
              HDData<T> d_alpha(alpha);
              MatrixSum(a, b, d_alpha(true), c);
          });
    m.def("VectorSum",
          [](D_Array &a, D_Array &b, T alpha) {
              D_Array c(a.n);
              HDData<T> d_alpha(alpha);
              VectorSum(a, b, d_alpha(true), c);
              return std::move(c);
          },
          py::return_value_policy::move);
    m.def("VectorSum",
          [](D_Array &a, D_Array &b) {
              D_Array c(a.n);
              VectorSum(a, b, c);
              return std::move(c);
          },
          py::return_value_policy::move);
}