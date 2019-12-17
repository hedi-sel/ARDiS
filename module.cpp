#include <pybind11/pybind11.h>

#include "main.h"

PYBIND11_MODULE(myModule, m)
{
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("launch", &launch, "A function which adds two numbers");
}