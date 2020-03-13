#pragma once

#include <cuda_runtime.h>

#include "constants.hpp"
#include "dataStructures/array.hpp"
#include "dataStructures/hd_data.hpp"
#include "hediHelper/cuda/cuda_thread_manager.hpp"

enum OpType { sum, maximum };

T ReductionOperation(D_Array &A, OpType op);

T ReductionIncreasing(int *A, int n);