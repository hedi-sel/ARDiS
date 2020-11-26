#pragma once
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <iostream>
#include <string>

#define CUSOLVER_ERROR_SIZE 12
extern const char *cusolverErrors[CUSOLVER_ERROR_SIZE];

#define cusolverErrchk(ans)                                                    \
    { cusolverAssert((ans), __FILE__, __LINE__); }

inline void cusolverAssert(cusolverStatus_t code, const char *file, int line,
                           bool abort = true) {
    if (code != CUSOLVER_STATUS_SUCCESS) {
        assert(code < CUSOLVER_ERROR_SIZE);
        std::cout << "GPUassert: Error " << cusolverErrors[code] << " at "
                  << file << "(" << line << ")\n";
        if (abort)
            exit(code);
    }
}