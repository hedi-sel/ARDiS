#pragma once

#include <cuda_runtime.h>

#include "constants.hpp"
#include "dataStructures/array.hpp"
#include "dataStructures/hd_data.hpp"
#include "hediHelper/cuda/cuda_thread_manager.hpp"

T ReductionIncreasing(int *A, int n);