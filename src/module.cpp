#include <pybind11/pybind11.h>

#include "main.hpp"

PYBIND11_MODULE(dna, m) {
    m.doc() = "Sparse Linear Equation solving API"; // optional module docstring
    m.def("SolveLinEq", &SolveLinEq, "SolveLinEq linear system resolution");
    m.def("LoadMatrixFromFile", &LoadMatrixFromFile, py::arg("path"),
          py::arg("sendToGPU") = true,
          "Reand and copy sparse format Matrix into GPU Memory");
    m.def("ConvertMatrixToCSR", &ConvertMatrixToCSR,
          "Reorder matrix rows and convert to CSR data format");
    m.def("PrintMatrix", &PrintMatrix, py::arg("printGpuVersion") = true,
          "Print the matrix as it is in GPU memory");
}