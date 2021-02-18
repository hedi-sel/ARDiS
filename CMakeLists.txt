cmake_minimum_required(VERSION 3.10)

set (CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_BIN_PATH}) 
set (CUDA_INCLUDE_DIRS $ENV{CUDA_BIN_PATH}/include) 
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CUDA_STANDARD 14)
project(PdeSolver LANGUAGES CUDA CXX)

set(SRC ${CMAKE_SOURCE_DIR}/src)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/pythonLib/ardis)
set(Python_ADDITIONAL_VERSIONS 3.6)

find_package(CUDA REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${PYBIND_PATH})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${SRC})
include_directories(${SRC}/include)

file(GLOB_RECURSE SRC_FILES "${SRC}/*.cu" "${SRC}/*.cpp")

add_library(ardisLib SHARED ${SRC_FILES})
set_target_properties(ardisLib PROPERTIES PREFIX "")
set_target_properties(ardisLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS --extended-lambda)

target_link_libraries(ardisLib ${PYTHON_LIBRARIES})
target_link_libraries(ardisLib cudart cusolver cusparse cublas)
