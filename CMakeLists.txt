cmake_minimum_required(VERSION 3.10)

project(PdeSolver LANGUAGES CUDA CXX)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CUDA_STANDARD 14)

set(SRC ${CMAKE_SOURCE_DIR}/src)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${SRC}/modulePython)
# set(CMAKE_CUDA_FLAGS "-dc")

find_package(CUDA)
find_package(PythonLibs 3.7 REQUIRED)
find_package(pybind11)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${SRC})
include_directories(${SRC}/include)

file(GLOB_RECURSE SRC_FILES "${SRC}/*.cpp" "${SRC}/*.cu")

message(${SRC_FILES})

add_library(dna SHARED ${SRC_FILES})
set_target_properties(dna PROPERTIES PREFIX "")
set_target_properties(dna PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(dna ${PYTHON_LIBRARIES})
target_link_libraries(dna cudart cusolver cusparse)
target_link_libraries(dna ${CUDA_LIBRARIES})

# set(CMAKE_VERBOSE_MAKEFILE true)