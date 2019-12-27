#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#define gpuErrchk(ans)                                                         \
    { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        std::cout << "GPUassert: Error " << code << " at " << file << " "
                  << line << "\n";
        if (abort)
            exit(code);
    }
}