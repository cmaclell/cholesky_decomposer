cmake_minimum_required(VERSION 3.15...3.29)
project(cholesky_decomposer LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

pybind11_add_module(cholesky_decomposer cholesky_decomposer.cpp)
target_link_libraries(cholesky_decomposer PRIVATE Eigen3::Eigen)

install(TARGETS cholesky_decomposer DESTINATION .)
