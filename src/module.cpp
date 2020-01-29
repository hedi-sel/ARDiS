#include "pybind11_include.hpp"
#include <assert.h>

#include "dataStructures/array.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "main.h"
#include "matrixOperations/basic_operations.hpp"

PYBIND11_MODULE(dna, m) {
    py::enum_<MatrixType>(m, "MatrixType")
        .value("COO", COO)
        .value("CSR", CSR)
        .value("CSC", CSC)
        .export_values();

    py::class_<State>(m, "State")
        .def(py::init<int>())
        .def("AddSpecies", &State::AddSpecies,
             py::return_value_policy::take_ownership)
        .def("GetSpecies", &State::GetSpecies,
             py::return_value_policy::take_ownership)
        .def("Print", &State::Print, py::arg("printCount") = 5);

    py::class_<System>(m, "System")
        .def(py::init<int>())
        .def("IterateDiffusion", &System::IterateDiffusion)
        .def("IterateReaction", &System::IterateReaction)
        .def(
            "AddSpecies",
            [](System &self, std::string name) { self.state.AddSpecies(name); })
        .def("SetSpecies",
             [](System &self, std::string name, D_Array &sub_state) {
                 self.state.GetSpecies(name) = sub_state;
             })
        .def("SetSpecies",
             [](System &self, std::string name, py::array_t<T> &sub_state) {
                 gpuErrchk(cudaMemcpy(
                     self.state.GetSpecies(name).vals, sub_state.data(),
                     sizeof(T) * sub_state.size(), cudaMemcpyHostToDevice));
             })
        .def("LoadDampnessMatrix", &System::LoadDampnessMatrix)
        .def("LoadStiffnessMatrix", &System::LoadStiffnessMatrix)
        .def("Print", &System::Print, py::arg("printCount") = 5)
        .def_readwrite("State", &System::state,
                       py::return_value_policy::take_ownership);

    py::class_<D_SparseMatrix>(m, "D_SparseMatrix")
        .def(py::init<int, int, int, MatrixType>())
        .def(py::init<int, int, int>())
        .def(py::init<int, int>())
        .def(py::init<>())
        .def(py::init<const D_SparseMatrix &, bool>())
        .def(py::init<const D_SparseMatrix &>())
        .def("AddElement", &D_SparseMatrix::AddElement)
        .def("ConvertMatrixToCSR", &D_SparseMatrix::ConvertMatrixToCSR)
        .def("Print", &D_SparseMatrix::Print, py::arg("printCount") = 5)
        .def("Dot",
             [](D_SparseMatrix &mat, D_Array &x) {
                 D_Array y(mat.rows);
                 Dot(mat, x, y);
                 return std::move(y);
             },
             py::return_value_policy::move)
        .def("__imul__",
             [](D_SparseMatrix &self, T alpha) {
                 HDData<T> d_alpha(-alpha);
                 ScalarMult(self, d_alpha(true));
                 return self;
             },
             py::return_value_policy::take_ownership)
        .def_readonly("nnz", &D_SparseMatrix::nnz)
        .def_readonly("rows", &D_SparseMatrix::rows)
        .def_readonly("cols", &D_SparseMatrix::cols)
        .def_readonly("type", &D_SparseMatrix::type)
        .def_readonly("isDevice", &D_SparseMatrix::isDevice);

    py::class_<D_Array>(m, "D_Array")
        .def(py::init<int>())
        .def(py::init<const D_Array &>())
        .def("Print", &D_Array::Print, py::arg("printCount") = 5)
        .def("Norm", &D_Array::Norm)
        .def("Fill",
             [](D_Array &self, py::array_t<T> &x) {
                 assert(x.size() == self.n);
                 gpuErrchk(cudaMemcpy(self.vals, x.data(), sizeof(T) * x.size(),
                                      cudaMemcpyHostToDevice));
             })
        .def("Dot",
             [](D_Array &self, D_Array &b) {
                 HDData<T> res;
                 Dot(self, b, res(true));
                 res.SetHost();
                 return res();
             })
        .def("ToNumpyArray",
             [](D_Array &self) {
                 T *data = new T[self.n];
                 cudaMemcpy(data, self.vals, sizeof(T) * self.n,
                            cudaMemcpyDeviceToHost);
                 return py::array_t(self.n, data);
             })
        .def("__add__",
             [](D_Array &self, D_Array &b) {
                 D_Array c(self.n);
                 VectorSum(self, b, c);
                 return std::move(c);
             },
             py::return_value_policy::take_ownership)
        .def("__sub__",
             [](D_Array &self, D_Array &b) {
                 D_Array c(self.n);
                 HDData<T> m1(-1);
                 VectorSum(self, b, m1(true), c);
                 return std::move(c);
             },
             py::return_value_policy::take_ownership)
        .def("__imul__",
             [](D_Array &self, T alpha) {
                 HDData<T> d_alpha(-alpha);
                 ScalarMult(self, d_alpha(true));
                 return self;
             },
             py::return_value_policy::take_ownership);

    m.doc() = "Sparse Linear Equation solving API"; // optional module docstring
    m.def("SolveCholesky", &SolveCholesky,
          py::return_value_policy::take_ownership);
    m.def("SolveConjugateGradient", &SolveConjugateGradient,
          py::return_value_policy::take_ownership);
    m.def("ReadFromFile", &ReadFromFile,
          py::return_value_policy::take_ownership);
    m.def("DiffusionTest", &DiffusionTest,
          py::return_value_policy::take_ownership);
    m.def("Test", &Test, py::return_value_policy::move);

    m.def("GetNumpyVector",
          [](D_Array &v) {
              D_Array x(v, true);
              return std::move(py::array_t(x.n, x.vals));
          },
          py::return_value_policy::take_ownership);
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
          py::return_value_policy::take_ownership);
    m.def("VectorSum",
          [](D_Array &a, D_Array &b) {
              D_Array c(a.n);
              VectorSum(a, b, c);
              return std::move(c);
          },
          py::return_value_policy::take_ownership);
} // namespace PYBIND11_MODULE(dna,m)