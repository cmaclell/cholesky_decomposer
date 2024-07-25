cmake_minimum_required(VERSION 3.15...3.29)
project(cholsky_update LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(cholsky_update cholsky_eigen.cpp)
install(TARGET cholsky_update DESTINATION .)
