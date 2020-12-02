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
#include "reactionDiffusionSystem/simulation.hpp"

PYBIND11_MODULE(ardisLib, m) {
    py::enum_<matrix_type>(m, "matrix_type")
        .value("COO", COO)
        .value("CSR", CSR)
        .value("CSC", CSC)
        .export_values();

    py::class_<state>(m, "state")
        .def(py::init<int>())
        .def("add_species", &state::add_species,
             py::return_value_policy::reference)
        .def("get_species", &state::get_species,
             py::return_value_policy::reference)
        .def("print", &state::print, py::arg("printCount") = 5)
        .def("list_species",
             [](state &self) {
                 py::list listSpecies;
                 for (std::map<std::string, int>::iterator it =
                          self.names.begin();
                      it != self.names.end(); ++it) {
                     listSpecies.append(it->first);
                 }
                 return listSpecies;
             })
        .def("__len__", &state::n_species)
        .def("vector_size", &state::size);
    py::class_<simulation>(m, "simulation")
        .def(py::init<int>())
        .def("iterate_diffusion", &simulation::iterate_diffusion)
        .def("iterate_reaction", &simulation::iterate_reaction)
        .def("prune", &simulation::prune, py::arg("value") = 0)
        .def("add_species",
             [](simulation &self, std::string name) {
                 self.current_state.add_species(name);
             })
        .def("set_species",
             [](simulation &self, std::string name, d_vector &sub_state) {
                 assert(sub_state.size() == self.current_state.size());
                 gpuErrchk(cudaMemcpy(
                     self.current_state.get_species(name).data, sub_state.data,
                     sizeof(T) * sub_state.size(), cudaMemcpyDeviceToDevice));
             })
        .def("set_species",
             [](simulation &self, std::string name, py::array_t<T> &sub_state) {
                 assert(sub_state.size() == self.current_state.size());
                 gpuErrchk(cudaMemcpy(self.current_state.get_species(name).data,
                                      sub_state.data(),
                                      sizeof(T) * sub_state.size(),
                                      cudaMemcpyHostToDevice));
             })
        .def("add_reaction", static_cast<void (simulation::*)(
                                 std::string, int, std::string, int, T)>(
                                 &simulation::add_reaction))
        .def("add_reaction",
             static_cast<void (simulation::*)(const std::string &, T)>(
                 &simulation::add_reaction))
        .def("add_mm_reaction",
             static_cast<void (simulation::*)(const std::string &, T, T)>(
                 &simulation::add_mm_reaction))
        .def("add_mm_reaction",
             static_cast<void (simulation::*)(std::string, std::string, int, T,
                                              T)>(&simulation::add_mm_reaction))
        .def(
            "get_species",
            [](simulation &self, std::string name) {
                return &self.current_state.get_species(name);
            },
            py::return_value_policy::reference)
        .def(
            "get_diffusion_matrix",
            [](simulation &self) { return self.diffusion_matrix; },
            py::return_value_policy::reference)
        .def(
            "get_damping_matrix",
            [](simulation &self) { return *self.damp_mat; },
            py::return_value_policy::reference)
        .def(
            "get_stiffness_matrix",
            [](simulation &self) { return *self.stiff_mat; },
            py::return_value_policy::reference)
        .def("load_dampness_matrix", &simulation::load_dampness_matrix)
        .def("load_stiffness_matrix", &simulation::load_stiffness_matrix)
        .def("print", &simulation::print, py::arg("print_count") = 5)
        .def_readwrite("state", &simulation::current_state,
                       py::return_value_policy::reference)
        .def_property(
            "epsilon",
            [](simulation &self) { // Getter
                return self.epsilon;
            },
            [](simulation &self, T value) { // Setter
                self.epsilon = value;
            })
        .def_property(
            "drain",
            [](simulation &self) { // Getter
                return self.drain;
            },
            [](simulation &self, T value) { // Setter
                self.drain = value;
            });

    py::class_<d_spmatrix>(m, "d_spmatrix")
        .def(py::init<int, int, int, matrix_type>())
        .def(py::init<int, int, int>())
        .def(py::init<int, int>())
        .def(py::init<>())
        .def(py::init<const d_spmatrix &, bool>())
        .def(py::init<const d_spmatrix &>())
        .def(py::init([](int n_rows, int n_cols, py::array_t<int> &rowArr,
                         py::array_t<int> &colArr, py::array_t<T> &data,
                         matrix_type type = COO) {
            d_spmatrix self = d_spmatrix(n_rows, n_cols, data.size(), type);
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
        .def("to_csr", &d_spmatrix::to_csr)
        .def("print", &d_spmatrix::print, py::arg("print_count") = 5)
        .def(
            "dot",
            [](d_spmatrix &mat, d_vector &x) {
                d_vector y(mat.rows);
                dot(mat, x, y);
                return std::move(y);
            },
            py::return_value_policy::move)
        .def("__eq__", &d_spmatrix::operator==)
        .def(
            "__imul__",
            [](d_spmatrix &self, T alpha) {
                hd_data<T> d_alpha(-alpha);
                scalar_mult(self, d_alpha(true));
                return &self;
            },
            py::return_value_policy::take_ownership)
        .def(
            "__mul__",
            [](d_spmatrix &self, T alpha) {
                hd_data<T> d_alpha(-alpha);
                d_spmatrix self_copy(self);
                scalar_mult(self_copy, d_alpha(true));
                return self_copy;
            },
            py::return_value_policy::take_ownership)
        .def(
            "__add__",
            [](d_spmatrix &self, d_spmatrix &b) {
                d_spmatrix *c = new d_spmatrix();
                matrix_sum(self, b, *c);
                return c;
            },
            py::return_value_policy::take_ownership)
        .def(
            "__sub__",
            [](d_spmatrix &self, d_spmatrix &b) {
                d_spmatrix *c = new d_spmatrix();
                hd_data<T> m1(-1);
                matrix_sum(self, b, m1(true), *c);
                return c;
            },
            py::return_value_policy::take_ownership)
        .def("__str__", &d_spmatrix::ToString)
        .def("__len__", [](d_spmatrix &self) { return self.nnz; })
        .def_property_readonly("shape",
                               [](d_spmatrix &self) {
                                   return py::make_tuple(self.rows, self.cols);
                               })
        .def_readonly("nnz", &d_spmatrix::nnz)
        .def_readonly("dtype", &d_spmatrix::type);

    py::class_<d_vector>(m, "d_vector")
        .def(py::init<int>())
        .def(py::init<const d_vector &>())
        .def(py::init([](py::array_t<T> &x) {
            auto vector = d_vector(x.size());
            gpuErrchk(cudaMemcpy(vector.data, x.data(), sizeof(T) * x.size(),
                                 cudaMemcpyHostToDevice));
            return std::move(vector);
        }))
        .def("at", &d_vector::at)
        .def("print", &d_vector::print, py::arg("printCount") = 5)
        .def("norm",
             [](d_vector &self) {
                 hd_data<T> norm;
                 dot(self, self, norm(true));
                 norm.SetHost();
                 return sqrt(norm());
             })
        .def("fill_value", &d_vector::Fill)
        .def("prune", &d_vector::prune, py::arg("value") = 0)
        .def("prune_under", &d_vector::prune_under, py::arg("value") = 0)
        .def("dot",
             [](d_vector &self, d_vector &b) {
                 hd_data<T> res;
                 dot(self, b, res(true));
                 res.SetHost();
                 return res();
             })
        .def("toarray",
             [](d_vector &self) {
                 T *data = new T[self.n];
                 cudaMemcpy(data, self.data, sizeof(T) * self.n,
                            cudaMemcpyDeviceToHost);
                 return py::array_t(self.n, data);
             })
        .def(
            "__add__",
            [](d_vector &self, d_vector &b) {
                d_vector c(self.n);
                vector_sum(self, b, c);
                return std::move(c);
            },
            py::return_value_policy::take_ownership)
        .def(
            "__sub__",
            [](d_vector &self, d_vector &b) {
                d_vector c(self.n);
                hd_data<T> m1(-1);
                vector_sum(self, b, m1(true), c);
                return std::move(c);
            },
            py::return_value_policy::take_ownership)
        .def("__imul__",
             [](d_vector &self, T alpha) {
                 hd_data<T> d_alpha(-alpha);
                 scalar_mult(self, d_alpha(true));
                 return self;
             })
        .def("__len__", &d_vector::size)
        .def("__str__", &d_vector::ToString);

    m.doc() = "Sparse Linear Equation solving API"; // optional module docstring

    ///////
    /// Deprecated
    ///
    // m.def("solve_cholesky", &solve_cholesky,
    //       py::return_value_policy::take_ownership);
    // m.def("SolveConjugateGradientRawData",
    //       [](d_spmatrix *d_mat, d_vector *b, d_vector *x, T epsilon) {
    //           return cg_solver::StaticCGSolve(*d_mat, *b, *x, epsilon);
    //       });
    // m.def("read_file", &read_file, py::return_value_policy::take_ownership);

    m.def("matrix_sum", [](d_spmatrix &a, d_spmatrix &b, d_spmatrix &c) {
        matrix_sum(a, b, c);
    });
    m.def("matrix_sum",
          [](d_spmatrix &a, d_spmatrix &b, T alpha, d_spmatrix &c) {
              hd_data<T> d_alpha(alpha);
              matrix_sum(a, b, d_alpha(true), c);
          });
    m.def("ToCSV", [](state &state, const std::string &path) {
        return ToCSV(state, path);
    });
    m.def("ToCSV", [](d_vector &array, std::string &path) {
        return ToCSV(array, path);
    });
    m.def("ToCSV", [](py::array_t<T> &array, std::string &path) {
        d_vector arrayContainer(array.size(), false);
        cudaMemcpy(arrayContainer.data, array.data(),
                   sizeof(T) * arrayContainer.n, cudaMemcpyHostToHost);
        return ToCSV(arrayContainer, path);
    });

    py::module geometry = m.def_submodule("geometry");

    py::class_<zone>(geometry, "zone");
    py::class_<rect_zone, zone>(geometry, "rect_zone")
        .def(py::init<>())
        .def(py::init<float, float, float, float>())
        .def(py::init<point2d, point2d>())
        .def("is_inside",
             static_cast<bool (rect_zone::*)(T, T)>(&rect_zone::is_inside))

        .def("is_inside",
             static_cast<bool (rect_zone::*)(point2d)>(&rect_zone::is_inside));
    py::class_<tri_zone, zone>(geometry, "tri_zone")
        .def(py::init<>())
        .def(py::init<float, float, float, float, float, float>())
        .def(py::init<point2d, point2d, point2d>())
        .def("is_inside",
             static_cast<bool (tri_zone::*)(T, T)>(&tri_zone::is_inside))
        .def("is_inside",
             static_cast<bool (tri_zone::*)(point2d)>(&tri_zone::is_inside));

    py::class_<point2d>(geometry, "point2d")
        .def(py::init<T, T>())
        .def(py::init<>());

    py::module d_geometry = m.def_submodule("d_geometry");

    d_geometry.def("fill_zone", &fill_zone);
    d_geometry.def("fill_outside_zone", &fill_outside_zone);
    d_geometry.def("GetMinZone", &GetMinZone);
    d_geometry.def("GetMaxZone", &GetMaxZone);
    d_geometry.def("GetMeanZone", &GetMeanZone);

    py::class_<d_mesh>(d_geometry, "d_mesh")
        .def(py::init<d_vector &, d_vector &>())
        .def(py::init([](py::array_t<T> &x, py::array_t<T> &y) {
                 assert(x.size() == y.size());
                 auto mesh = d_mesh(x.size());
                 gpuErrchk(cudaMemcpy(mesh.X.data, x.data(),
                                      sizeof(T) * x.size(),
                                      cudaMemcpyHostToDevice));
                 gpuErrchk(cudaMemcpy(mesh.Y.data, y.data(),
                                      sizeof(T) * x.size(),
                                      cudaMemcpyHostToDevice));
                 return std::move(mesh);
             }),
             py::return_value_policy::move)
        .def("__len__", &d_mesh::size)
        .def_readonly("X", &d_mesh::X)
        .def_readonly("Y", &d_mesh::Y);

} // namespace PYBIND11_MODULE(dna,m)