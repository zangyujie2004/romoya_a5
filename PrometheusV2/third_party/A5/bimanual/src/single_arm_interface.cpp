#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "arx_r5_src/interfaces/InterfacesPy.hpp"

namespace py = pybind11;

PYBIND11_MODULE(arx_r5_python, m) {
    py::class_<arx::r5::InterfacesPy>(m, "InterfacesPy")
        .def(py::init<std::string,std::string>()) 
        .def("set_joint_positions", &arx::r5::InterfacesPy::set_joint_positions)
        .def("set_ee_pose", &arx::r5::InterfacesPy::set_ee_pose)
        .def("set_arm_status", &arx::r5::InterfacesPy::set_arm_status)
        .def("set_catch", &arx::r5::InterfacesPy::set_catch)
        .def("mit_joint_control", &arx::r5::InterfacesPy::mit_joint_control)
        .def("get_joint_positions", &arx::r5::InterfacesPy::get_joint_positions)
        .def("get_joint_velocities", &arx::r5::InterfacesPy::get_joint_velocities)
        .def("get_joint_currents", &arx::r5::InterfacesPy::get_joint_currents)
        .def("get_joint_torques", &arx::r5::InterfacesPy::get_joint_torques)
        .def("get_ext_wrench", &arx::r5::InterfacesPy::get_ext_wrench)
        .def("set_ext_wrench", &arx::r5::InterfacesPy::set_ext_wrench)
        .def("get_gripper_ext_torque", &arx::r5::InterfacesPy::get_gripper_ext_torque)
        .def("set_gripper_ext_torque", &arx::r5::InterfacesPy::set_gripper_ext_torque)
        .def("set_slave_gripper_position", &arx::r5::InterfacesPy::set_slave_gripper_position)
        .def("arx_x", &arx::r5::InterfacesPy::arx_x)
        .def("get_ee_pose", &arx::r5::InterfacesPy::get_ee_pose);
}
