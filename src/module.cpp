#include <pybind11/pybind11.h>

#include "main.hpp"

PYBIND11_MODULE(dna, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("launch", &launch, "Run some code");
}