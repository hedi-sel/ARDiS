cmake_minimum_required(VERSION 3.10)

set (CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_BIN_PATH}) 
set (CUDA_INCLUDE_DIRS $ENV{CUDA_BIN_PATH}/include) 
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CUDA_STANDARD 14)
project(PdeSolver LANGUAGES CUDA CXX)

set(SRC ${CMAKE_SOURCE_DIR}/src)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${SRC}/pythonScripts/modulePython)
set(BOOST_ROOT $ENV{BOOST_ROOT})
# set(CMAKE_CUDA_FLAGS "-dc")

find_package(CUDA)
find_package(PythonLibs 3.7 REQUIRED)
find_package(pybind11)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
message(${CUDA_INCLUDE_DIRS})
include_directories(${BOOST_ROOT})
include_directories(${SRC})
include_directories(${SRC}/include)
link_directories(${BOOST_ROOT}/stage/lib)

file(GLOB_RECURSE SRC_FILES "${SRC}/*.cu" "${SRC}/*.cpp")

message(${SRC_FILES})

add_library(dna SHARED ${SRC_FILES})
set_target_properties(dna PROPERTIES PREFIX "")
set_target_properties(dna PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS --extended-lambda)

target_link_libraries(dna ${PYTHON_LIBRARIES})
target_link_libraries(dna cudart cusolver cusparse)
# target_link_libraries(dna boost_timer)

# set(CMAKE_VERBOSE_MAKEFILE true)