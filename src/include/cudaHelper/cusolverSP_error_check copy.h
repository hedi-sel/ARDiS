#pragma once
#include <cuda_runtime.h>
#include <cusolverSp.h>

#define cusolverErrchk(ans)                                                    \
    { cusolverAssert((ans), __FILE__, __LINE__); }

inline void cusolverAssert(cusolverStatus_t code, const char *file, int line,
                           bool abort = true) {
    if (code != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "GPUassert: Error%i %s %d\n", code, file, line);
        if (abort)
            exit(code);
    }
}