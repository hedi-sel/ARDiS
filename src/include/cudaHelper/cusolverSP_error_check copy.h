#pragma once
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <iostream>

#define cusolverErrchk(ans)                                                    \
    { cusolverAssert((ans), __FILE__, __LINE__); }

inline void cusolverAssert(cusolverStatus_t code, const char *file, int line,
                           bool abort = true) {
    if (code != CUSOLVER_STATUS_SUCCESS) {
        std::cout << "GPUassert: Error " << code << " " << file << " " << line
                  << "\n";
        if (abort)
            exit(code);
    }
}