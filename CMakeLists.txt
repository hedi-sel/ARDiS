cmake_minimum_required(VERSION 3.10)

project(testPybinb CXX CUDA)

add_subdirectory(pybind11)

pybind11_add_module(app test.cpp)