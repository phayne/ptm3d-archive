#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../../cpp/illumination/illumination_model.h"

namespace py = pybind11;
using namespace europa;

PYBIND11_MODULE(europa_illumination, m) {
    m.doc() = "Python bindings for Europa illumination model";
    
    py::class_<IlluminationModel>(m, "IlluminationModel")
        .def(py::init<>())
        .def("set_time_steps", &IlluminationModel::setTimeSteps)
        .def("initialize_spice", [](IlluminationModel& self) {
            return self.initializeSpice();
        })
        .def("convert_time_to_et", [](IlluminationModel& self, const std::string& time_str) {
            return self.convertTimeToET(time_str);
        })
        .def("get_sun_position", [](IlluminationModel& self, double et) {
            return self.getSunPosition(et);
        })
        .def("get_jupiter_position", [](IlluminationModel& self, double et) {
            return self.getJupiterPosition(et);
        })
        .def("get_times", [](const IlluminationModel& self) {
            return self.getTimes();
        })
        .def("get_time_strings", [](const IlluminationModel& self) {
            return self.getTimeStrings();
        })
        .def("get_sun_positions", [](const IlluminationModel& self) {
            return self.getSunPositions();
        })
        .def("get_jupiter_positions", [](const IlluminationModel& self) {
            return self.getJupiterPositions();
        })
        .def("cartesian_from_latlon", 
            [](double lat, double lon, double radius = 1.0) {
                return IlluminationModel::cartesianFromLatLon(lat, lon, radius);
            })
        .def("latlon_from_cartesian", 
            [](double x, double y, double z) {
                return IlluminationModel::latLonFromCartesian(x, y, z);
            });
}