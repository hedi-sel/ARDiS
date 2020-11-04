#include "pybind11_include.hpp"
#include <assert.h>
#include <math.h>
#include <vector>

#include "dataStructures/array.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "dataStructures/zoneManager/rectangle_zone.hpp"
#include "dataStructures/zoneManager/zone_methods.hpp"
#include "main.h"
#include "matrixOperations/basic_operations.hpp"
#include "reactionDiffusionSystem/system.hpp"

PYBIND11_MODULE(dna, m) {
    py::enum_<MatrixType>(m, "MatrixType")
        .value("COO", COO)
        .value("CSR", CSR)
        .value("CSC", CSC)
        .export_values();

    py::class_<State>(m, "State")
        .def(py::init<int>())
        .def("AddSpecies", &State::AddSpecies,
             py::return_value_policy::reference)
        .def("GetSpecies", &State::GetSpecies,
             py::return_value_policy::reference)
        .def("Print", &State::Print, py::arg("printCount") = 5);
    py::class_<System>(m, "System")
        .def(py::init<int>())
        .def("IterateDiffusion", &System::IterateDiffusion)
        .def("IterateReaction", &System::IterateReaction)
        .def("Prune", &System::Prune, py::arg("value") = 0)
        .def(
            "AddSpecies",
            [](System &self, std::string name) { self.state.AddSpecies(name); })
        .def("SetSpecies",
             [](System &self, std::string name, D_Array sub_state) {
                 self.state.GetSpecies(name) = sub_state;
             })
        .def("SetSpecies",
             [](System &self, std::string name, py::array_t<T> &sub_state) {
                 gpuErrchk(cudaMemcpy(
                     self.state.GetSpecies(name).vals, sub_state.data(),
                     sizeof(T) * sub_state.size(), cudaMemcpyHostToDevice));
                 self.state.GetSpecies(name).Print();
             })
        .def("AddReaction",
             [](System &self, std::string reag, int kr, std::string prod,
                int kp, T rate) {
                 std::vector<stochCoeff> input;
                 std::vector<stochCoeff> output;
                 input.push_back(std::pair(reag, kr));
                 output.push_back(std::pair(prod, kp));
                 self.AddReaction(ReactionMassAction(input, output, rate));
             })
        .def("AddReaction", [](System &self, std::string reaction,
                               T rate) { self.AddReaction(reaction, rate); })
        .def("AddMMReaction",
             [](System &self, std::string reaction, T Vm, T Km) {
                 self.AddMMReaction(reaction, Vm, Km);
             })
        .def("AddMMReaction",
             [](System &self, std::string reag, std::string prod, int kp, T Vm,
                T Km) { self.AddMMReaction(reag, prod, kp, Vm, Km); })
        .def(
            "GetSpecies",
            [](System &self, std::string name) {
                return &self.state.GetSpecies(name);
            },
            py::return_value_policy::reference)
        .def(
            "GetDiffusionMatrix",
            [](System &self) { return self.diffusion_matrix; },
            py::return_value_policy::reference)
        .def(
            "GetDampingMatrix", [](System &self) { return *self.damp_mat; },
            py::return_value_policy::reference)
        .def(
            "GetStiffnessMatrix", [](System &self) { return *self.stiff_mat; },
            py::return_value_policy::reference)
        .def("LoadDampnessMatrix", &System::LoadDampnessMatrix)
        .def("LoadStiffnessMatrix", &System::LoadStiffnessMatrix)
        .def("Print", &System::Print, py::arg("printCount") = 5)
        .def_readwrite("State", &System::state,
                       py::return_value_policy::reference)
        .def("SetDrain", &System::SetDrain)
        .def("SetEpsilon", &System::SetEpsilon);

    py::class_<D_SparseMatrix>(m, "D_SparseMatrix")
        .def(py::init<int, int, int, MatrixType>())
        .def(py::init<int, int, int>())
        .def(py::init<int, int>())
        .def(py::init<>())
        .def(py::init<const D_SparseMatrix &, bool>())
        .def(py::init<const D_SparseMatrix &>())
        .def(
            "Fill",
            [](D_SparseMatrix &self, py::array_t<int> &rows,
               py::array_t<int> &cols, py::array_t<T> &data) {
                assert(data.size() == self.nnz);
                if (self.type == CSR)
                    assert(rows.size() == self.rows + 1);
                else
                    assert(rows.size() == self.nnz);
                if (self.type == CSC)
                    assert(cols.size() == self.cols + 1);
                else
                    assert(cols.size() == self.nnz);
                gpuErrchk(cudaMemcpy(self.vals, data.data(),
                                     sizeof(T) * data.size(),
                                     cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(self.colPtr, cols.data(),
                                     sizeof(int) * cols.size(),
                                     cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(self.rowPtr, rows.data(),
                                     sizeof(int) * rows.size(),
                                     cudaMemcpyHostToDevice));
            },
            py::arg("rows"), py::arg("cols"), py::arg("data"))
        .def("ConvertMatrixToCSR", &D_SparseMatrix::ConvertMatrixToCSR)
        .def("Print", &D_SparseMatrix::Print, py::arg("printCount") = 5)
        .def(
            "Dot",
            [](D_SparseMatrix &mat, D_Array &x) {
                D_Array y(mat.rows);
                Dot(mat, x, y);
                return std::move(y);
            },
            py::return_value_policy::move)
        .def("__eq__", &D_SparseMatrix::operator==)
        .def(
            "__imul__",
            [](D_SparseMatrix &self, T alpha) {
                HDData<T> d_alpha(-alpha);
                ScalarMult(self, d_alpha(true));
                return &self;
            },
            py::return_value_policy::take_ownership)
        .def(
            "__mul__",
            [](D_SparseMatrix &self, T alpha) {
                HDData<T> d_alpha(-alpha);
                D_SparseMatrix self_copy(self);
                ScalarMult(self_copy, d_alpha(true));
                return self_copy;
            },
            py::return_value_policy::take_ownership)
        .def(
            "__add__",
            [](D_SparseMatrix &self, D_SparseMatrix &b) {
                D_SparseMatrix *c = new D_SparseMatrix();
                MatrixSum(self, b, *c);
                return c;
            },
            py::return_value_policy::take_ownership)
        .def(
            "__sub__",
            [](D_SparseMatrix &self, D_SparseMatrix &b) {
                D_SparseMatrix *c = new D_SparseMatrix();
                HDData<T> m1(-1);
                MatrixSum(self, b, m1(true), *c);
                return c;
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
        .def("Norm",
             [](D_Array &self) {
                 HDData<T> norm;
                 Dot(self, self, norm(true));
                 norm.SetHost();
                 return sqrt(norm());
             })
        .def("Fill",
             [](D_Array &self, py::array_t<T> &x) {
                 assert(x.size() == self.n);
                 gpuErrchk(cudaMemcpy(self.vals, x.data(), sizeof(T) * x.size(),
                                      cudaMemcpyHostToDevice));
             })
        .def("FillValue", &D_Array::Fill)
        .def("Prune", &D_Array::Prune, py::arg("value") = 0)
        .def("PruneUnder", &D_Array::PruneUnder, py::arg("value") = 0)
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
        .def(
            "__add__",
            [](D_Array &self, D_Array &b) {
                D_Array c(self.n);
                VectorSum(self, b, c);
                return std::move(c);
            },
            py::return_value_policy::take_ownership)
        .def(
            "__sub__",
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
             })
        .def_readonly("n", &D_Array::n);

    py::class_<Zone>(m, "Zone")
        .def("IsInside",
             [](RectangleZone self, T x, T y) { return self.IsInside(x, y); })
        .def("IsInside",
             [](RectangleZone self, Point2D p) { return self.IsInside(p); });

    py::class_<RectangleZone, Zone>(m, "RectangleZone")
        .def(py::init<>())
        .def(py::init<float, float, float, float>())
        .def(py::init<Point2D, Point2D>());

    py::class_<Point2D>(m, "Point2D").def(py::init<T, T>()).def(py::init<>());

    m.doc() = "Sparse Linear Equation solving API"; // optional module docstring
    m.def("SolveCholesky", &SolveCholesky,
          py::return_value_policy::take_ownership);
    m.def("SolveConjugateGradientRawData",
          [](D_SparseMatrix *d_mat, D_Array *b, D_Array *x, T epsilon) {
              return CGSolver::StaticCGSolve(*d_mat, *b, *x, epsilon);
          });
    m.def("CppExplore", &LabyrinthExplore);
    m.def("ReadFromFile", &ReadFromFile,
          py::return_value_policy::take_ownership);
    m.def("MatrixSum", [](D_SparseMatrix &a, D_SparseMatrix &b,
                          D_SparseMatrix &c) { MatrixSum(a, b, c); });
    m.def("MatrixSum",
          [](D_SparseMatrix &a, D_SparseMatrix &b, T alpha, D_SparseMatrix &c) {
              HDData<T> d_alpha(alpha);
              MatrixSum(a, b, d_alpha(true), c);
          });
    m.def("ToCSV", [](State &state, const std::string &path) {
        return ToCSV(state, path);
    });
    m.def("ToCSV",
          [](D_Array &array, std::string &path) { return ToCSV(array, path); });
    m.def("ToCSV", [](py::array_t<T> &array, std::string &path) {
        D_Array arrayContainer(array.size(), false);
        cudaMemcpy(arrayContainer.vals, array.data(),
                   sizeof(T) * arrayContainer.n, cudaMemcpyHostToHost);
        return ToCSV(arrayContainer, path);
    });

    py::module zones = m.def_submodule("zones");
    zones.def("FillZone", &FillZone);
    zones.def("FillOutsideZone", &FillOutsideZone);
    zones.def("GetMinZone", &GetMinZone);
    zones.def("GetMaxZone", &GetMaxZone);
    zones.def("GetMeanZone", &GetMeanZone);
} // namespace PYBIND11_MODULE(dna,m)