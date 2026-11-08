#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/complex.h>
#include "admm_solver.h"

namespace py = pybind11;

PYBIND11_MODULE(lam_cpp, m) {
    m.doc() = "LAM fast ADMM solver (C++ with Eigen)";

    py::class_<lam::ADMMSolver>(m, "ADMMSolver")
        .def(py::init<>())
        .def("init", &lam::ADMMSolver::init,
             py::arg("A"), py::arg("cg_iters") = 10, py::arg("rho") = 100.0,
             "Initialize solver with steering matrix A (M x N, complex)")
        .def("solve", &lam::ADMMSolver::solve,
             py::arg("C"), py::arg("it") = 10, py::arg("tv") = 1.0,
             "Solve one frame: returns z1 (N,) acoustic intensity map")
        .def_readonly("N", &lam::ADMMSolver::N)
        .def_readonly("M", &lam::ADMMSolver::M);
}
