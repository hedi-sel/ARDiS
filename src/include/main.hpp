#include "pybind11_include.hpp"

py::array_t<double> SolveLinEq(py::array_t<double>);

void LoadMatrixFromFile(char *path, bool sendToGPU = true);

void SendMatrixToGpuMemory();

void ConvertMatrixToCSR();

void PrintMatrix(bool printGpuVersion = true);