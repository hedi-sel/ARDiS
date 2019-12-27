#include <pybind11/pybind11.h>

#include "main.hpp"
#include "sparseDataStruct/matrix_sparse.hpp"

PYBIND11_MODULE(dna, m) {
    m.doc() = "Sparse Linear Equation solving API"; // optional module docstring
    m.def("SolveLinEq", &SolveLinEq, "SolveLinEq linear system resolution");
    m.def("ReadFromFile", &ReadFromFile);

    py::enum_<MatrixType>(m, "MatrixType")
        .value("COO", COO)
        .value("CSR", CSR)
        .value("CSC", CSC)
        .export_values();

    py::class_<MatrixSparse>(m, "MatrixSparse")
        .def(py::init<int, int, int, MatrixType, bool>())
        .def(py::init<const MatrixSparse &, bool>())
        .def("AddElement", &MatrixSparse::AddElement)
        .def("ConvertMatrixToCSR", &MatrixSparse::ConvertMatrixToCSR)
        .def("Print", &MatrixSparse::Print);
}