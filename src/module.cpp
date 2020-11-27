#include "pybind11_include.hpp"
#include <assert.h>
#include <math.h>
#include <vector>

#include "dataStructures/array.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "geometry/mesh.hpp"
#include "geometry/zone.hpp"
#include "geometry/zone_methods.hpp"
#include "main.h"
#include "matrixOperations/basic_operations.hpp"
#include "reactionDiffusionSystem/system.hpp"

PYBIND11_MODULE(ardisLib, m) {
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
        .def("Print", &State::Print, py::arg("printCount") = 5)
        .def("ListSpecies",
             [](State &self) {
                 py::list listSpecies;
                 for (std::map<std::string, int>::iterator it =
                          self.names.begin();
                      it != self.names.end(); ++it) {
                     listSpecies.append(it->first);
                 }
                 return listSpecies;
             })
        .def("__len__", &State::n_species)
        .def("vector_size", &State::size);
    py::class_<System>(m, "System")
        .def(py::init<int>())
        .def("IterateDiffusion", &System::IterateDiffusion)
        .def("IterateReaction", &System::IterateReaction)
        .def("Prune", &System::Prune, py::arg("value") = 0)
        .def(
            "AddSpecies",
            [](System &self, std::string name) { self.state.AddSpecies(name); })
        .def("SetSpecies",
             [](System &self, std::string name, D_Vector sub_state) {
                 self.state.GetSpecies(name) = sub_state;
             })
        .def("SetSpecies",
             [](System &self, std::string name, py::array_t<T> &sub_state) {
                 gpuErrchk(cudaMemcpy(
                     self.state.GetSpecies(name).data, sub_state.data(),
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
        .def_property(
            "Epsilon",
            [](System &self) { // Getter
                return self.epsilon;
            },
            [](System &self, T value) { // Setter
                self.epsilon = value;
            })
        .def_property(
            "Drain",
            [](System &self) { // Getter
                return self.drain;
            },
            [](System &self, T value) { // Setter
                self.drain = value;
            });

    py::class_<D_SparseMatrix>(m, "D_SparseMatrix")
        .def(py::init<int, int, int, MatrixType>())
        .def(py::init<int, int, int>())
        .def(py::init<int, int>())
        .def(py::init<>())
        .def(py::init<const D_SparseMatrix &, bool>())
        .def(py::init<const D_SparseMatrix &>())
        .def(py::init([](int n_rows, int n_cols, py::array_t<int> &rowArr,
                         py::array_t<int> &colArr, py::array_t<T> &data,
                         MatrixType type = COO) {
            D_SparseMatrix self =
                D_SparseMatrix(n_rows, n_cols, data.size(), type);
            if (type == CSR)
                assert(rowArr.size() == self.rows + 1);
            else
                assert(rowArr.size() == self.nnz);
            if (type == CSC)
                assert(colArr.size() == self.cols + 1);
            else
                assert(colArr.size() == self.nnz);

            gpuErrchk(cudaMemcpy(self.data, data.data(),
                                 sizeof(T) * data.size(),
                                 cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(self.colPtr, colArr.data(),
                                 sizeof(int) * colArr.size(),
                                 cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(self.rowPtr, rowArr.data(),
                                 sizeof(int) * rowArr.size(),
                                 cudaMemcpyHostToDevice));
            return std::move(self);
        }))
        .def("ConvertMatrixToCSR", &D_SparseMatrix::ConvertMatrixToCSR)
        .def("Print", &D_SparseMatrix::Print, py::arg("printCount") = 5)
        .def(
            "Dot",
            [](D_SparseMatrix &mat, D_Vector &x) {
                D_Vector y(mat.rows);
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
        .def("__str__", &D_SparseMatrix::ToString)
        .def("__len__", [](D_SparseMatrix &self) { return self.nnz; })
        .def("shape",
             [](D_SparseMatrix &self) {
                 return py::make_tuple(self.rows, self.cols);
             })
        .def_readonly("Nnz", &D_SparseMatrix::nnz)
        .def_readonly("Rows", &D_SparseMatrix::rows)
        .def_readonly("Cols", &D_SparseMatrix::cols)
        .def_readonly("Type", &D_SparseMatrix::type)
        .def_readonly("IsDevice", &D_SparseMatrix::isDevice);

    py::class_<D_Vector>(m, "D_Vector")
        .def(py::init<int>())
        .def(py::init<const D_Vector &>())
        .def(py::init([](py::array_t<T> &x) {
            auto vector = D_Vector(x.size());
            gpuErrchk(cudaMemcpy(vector.data, x.data(), sizeof(T) * x.size(),
                                 cudaMemcpyHostToDevice));
            return std::move(vector);
        }))
        .def("at", &D_Vector::at)
        .def("Print", &D_Vector::Print, py::arg("printCount") = 5)
        .def("Norm",
             [](D_Vector &self) {
                 HDData<T> norm;
                 Dot(self, self, norm(true));
                 norm.SetHost();
                 return sqrt(norm());
             })
        .def("FillValue", &D_Vector::Fill)
        .def("Prune", &D_Vector::Prune, py::arg("value") = 0)
        .def("PruneUnder", &D_Vector::PruneUnder, py::arg("value") = 0)
        .def("Dot",
             [](D_Vector &self, D_Vector &b) {
                 HDData<T> res;
                 Dot(self, b, res(true));
                 res.SetHost();
                 return res();
             })
        .def("ToNumpyArray",
             [](D_Vector &self) {
                 T *data = new T[self.n];
                 cudaMemcpy(data, self.data, sizeof(T) * self.n,
                            cudaMemcpyDeviceToHost);
                 return py::array_t(self.n, data);
             })
        .def(
            "__add__",
            [](D_Vector &self, D_Vector &b) {
                D_Vector c(self.n);
                VectorSum(self, b, c);
                return std::move(c);
            },
            py::return_value_policy::take_ownership)
        .def(
            "__sub__",
            [](D_Vector &self, D_Vector &b) {
                D_Vector c(self.n);
                HDData<T> m1(-1);
                VectorSum(self, b, m1(true), c);
                return std::move(c);
            },
            py::return_value_policy::take_ownership)
        .def("__imul__",
             [](D_Vector &self, T alpha) {
                 HDData<T> d_alpha(-alpha);
                 ScalarMult(self, d_alpha(true));
                 return self;
             })
        .def("__len__", &D_Vector::size)
        .def("__str__", &D_Vector::ToString)
        .def_readonly("IsDevice", &D_Vector::isDevice);

    m.doc() = "Sparse Linear Equation solving API"; // optional module docstring
    m.def("SolveCholesky", &SolveCholesky,
          py::return_value_policy::take_ownership);
    m.def("SolveConjugateGradientRawData",
          [](D_SparseMatrix *d_mat, D_Vector *b, D_Vector *x, T epsilon) {
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
    m.def("ToCSV", [](D_Vector &array, std::string &path) {
        return ToCSV(array, path);
    });
    m.def("ToCSV", [](py::array_t<T> &array, std::string &path) {
        D_Vector arrayContainer(array.size(), false);
        cudaMemcpy(arrayContainer.data, array.data(),
                   sizeof(T) * arrayContainer.n, cudaMemcpyHostToHost);
        return ToCSV(arrayContainer, path);
    });

    py::module geometry = m.def_submodule("geometry");

    py::class_<Zone>(geometry, "Zone");
    py::class_<RectangleZone, Zone>(geometry, "RectangleZone")
        .def(py::init<>())
        .def(py::init<float, float, float, float>())
        .def(py::init<Point2D, Point2D>())
        .def("IsInside", [](RectangleZone &self, float x,
                            float y) { return self.IsInside(x, y); })
        .def("IsInside",
             [](RectangleZone &self, Point2D p) { return self.IsInside(p); });

    py::class_<TriangleZone, Zone>(geometry, "TriangleZone")
        .def(py::init<>())
        .def(py::init<float, float, float, float, float, float>())
        .def(py::init<Point2D, Point2D, Point2D>())
        .def("IsInside", [](TriangleZone &self, float x,
                            float y) { return self.IsInside(x, y); })
        .def("IsInside",
             [](TriangleZone &self, Point2D p) { return self.IsInside(p); });

    py::class_<Point2D>(geometry, "Point2D")
        .def(py::init<T, T>())
        .def(py::init<>());

    py::module d_geometry = m.def_submodule("d_geometry");

    d_geometry.def("FillZone", &FillZone);
    d_geometry.def("FillOutsideZone", &FillOutsideZone);
    d_geometry.def("GetMinZone", &GetMinZone);
    d_geometry.def("GetMaxZone", &GetMaxZone);
    d_geometry.def("GetMeanZone", &GetMeanZone);

    py::class_<D_Mesh>(d_geometry, "D_Mesh")
        .def(py::init<D_Vector &, D_Vector &>())
        .def(py::init([](py::array_t<T> x, py::array_t<T> y) {
            assert(x.size() == y.size());
            auto d_X = D_Vector(x.size());
            auto d_Y = D_Vector(y.size());
            gpuErrchk(cudaMemcpy(d_X.data, x.data(), sizeof(T) * x.size(),
                                 cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_Y.data, y.data(), sizeof(T) * x.size(),
                                 cudaMemcpyHostToDevice));
            return std::move(D_Mesh(d_X, d_Y));
        }))
        .def("__len__", &D_Mesh::size)
        .def_readonly("X", &D_Mesh::X)
        .def_readonly("Y", &D_Mesh::Y);

} // namespace PYBIND11_MODULE(dna,m)