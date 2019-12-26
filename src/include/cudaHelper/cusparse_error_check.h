#pragma once
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>

#define cusparseErrchk(ans)                                                    \
    { cusparseAssert((ans), __FILE__, __LINE__); }

inline void cusparseAssert(cusparseStatus_t code, const char *file, int line,
                           bool abort = true) {
    if (code != CUSPARSE_STATUS_SUCCESS) {
        std::cout << "GPUassert: Error" << code << file << line << "\n";
        if (abort)
            exit(code);
    }
}