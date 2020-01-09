#include <pybind11/pybind11.h>

#include "main.hpp"
#include "sparseDataStruct/matrix_sparse.hpp"

PYBIND11_MODULE(dna, m) {
    m.doc() = "Sparse Linear Equation solving API"; // optional module docstring
    m.def("SolveLinEq", &SolveLinEq, "SolveLinEq linear system resolution");
    m.def("ReadFromFile", &ReadFromFile);
    m.def("Test", &Test);

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
        .def("Print", &MatrixSparse::Print)
        .def_readwrite("n_elements", &MatrixSparse::n_elements)
        .def_readwrite("i_size", &MatrixSparse::i_size)
        .def_readwrite("j_size", &MatrixSparse::j_size)
        .def_readwrite("type", &MatrixSparse::type)
        .def_readonly("isDevice", &MatrixSparse::isDevice);
}