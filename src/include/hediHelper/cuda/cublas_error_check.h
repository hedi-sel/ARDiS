#pragma once
#include <assert.h>
#include <cublas.h>
#include <cuda_runtime.h>
#include <iostream>

#define CUBLAS_ERROR_SIZE 17
extern const char *cublasErrors[CUBLAS_ERROR_SIZE];

#define cublasErrchk(ans)                                                      \
    { cublasAssert((ans), __FILE__, __LINE__); }

inline void cublasAssert(cublasStatus_t code, const char *file, int line,
                         bool abort = true) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        assert(code < CUBLAS_ERROR_SIZE);
        std::cout << "GPUassert: Error " << cublasErrors[code] << " at " << file
                  << "(" << line << ")\n";
        if (abort)
            exit(code);
    }
}