#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "thermal_model.h"

namespace py = pybind11;

PYBIND11_MODULE(thermal_model_cpp, m) {
    m.doc() = "Europa thermal modeling module";
    
    // Bind the Config class
    py::class_<thermal::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("NYEARSEQ", &thermal::Config::NYEARSEQ)
        .def_readwrite("NPERDAY", &thermal::Config::NPERDAY)
        .def_readwrite("m", &thermal::Config::m)
        .def_readwrite("n", &thermal::Config::n)
        .def_readwrite("b", &thermal::Config::b)
        .def_readwrite("F", &thermal::Config::F)
        .def_readwrite("sigma", &thermal::Config::sigma)
        .def_readwrite("chi", &thermal::Config::chi)
        .def("getR350", &thermal::Config::getR350);

    // Bind the Planet class
    py::class_<thermal::Planet>(m, "Planet")
        .def(py::init<const std::string&>())
        .def_readwrite("name", &thermal::Planet::name)
        .def_readwrite("albedo", &thermal::Planet::albedo)
        .def_readwrite("emissivity", &thermal::Planet::emissivity)
        .def_readwrite("day", &thermal::Planet::day)
        .def_readwrite("year", &thermal::Planet::year)
        .def_readwrite("rAU", &thermal::Planet::rAU)
        .def_readwrite("S", &thermal::Planet::S)
        .def_readwrite("Qb", &thermal::Planet::Qb)
        .def_readwrite("ks", &thermal::Planet::ks)
        .def_readwrite("kd", &thermal::Planet::kd)
        .def_readwrite("rhos", &thermal::Planet::rhos)
        .def_readwrite("rhod", &thermal::Planet::rhod)
        .def_readwrite("H", &thermal::Planet::H)
        .def_readwrite("cp0", &thermal::Planet::cp0)
        .def_readwrite("cpCoeff", &thermal::Planet::cpCoeff)
        .def_readwrite("albedoCoef", &thermal::Planet::albedoCoef);
    
    // Bind the Profile class
    py::class_<thermal::Profile>(m, "Profile")
        .def(py::init<const thermal::Planet&, double, const thermal::Config&>())
        .def_readonly("z", &thermal::Profile::z)
        .def_readonly("T", &thermal::Profile::T)
        .def_readonly("k", &thermal::Profile::k)
        .def_readonly("cp", &thermal::Profile::cp);
    
    // Bind the Model class
    py::class_<thermal::Model>(m, "Model")
        .def(py::init<const thermal::Planet&, double, int, const Eigen::MatrixXd&, const thermal::Config&, bool>(),
             py::arg("planet"), py::arg("lat"), py::arg("ndays"), py::arg("flux_arr"), 
             py::arg("config"), py::arg("show_progress") = true)
        .def("run", &thermal::Model::run)
        .def("fastEquilibrate", &thermal::Model::fastEquilibrate)
        .def_readonly("profile", &thermal::Model::profile)
        .def_readonly("planet", &thermal::Model::planet)
        .def_readonly("T", &thermal::Model::T)
        .def_readonly("lt", &thermal::Model::lt)
        .def_readonly("N_steps", &thermal::Model::N_steps);
    
    // Helper functions
    m.def("skinDepth", &thermal::skinDepth);
    m.def("T_radeq", &thermal::T_radeq);
    m.def("T_eq", &thermal::T_eq);
}