#pragma once
#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>

#define CUSPARSE_ERROR_SIZE 10
extern const char *cusparseErrors[CUSPARSE_ERROR_SIZE];

#define cusparseErrchk(ans)                                                    \
    { cusparseAssert((ans), __FILE__, __LINE__); }

inline void cusparseAssert(cusparseStatus_t code, const char *file, int line,
                           bool abort = true) {
    if (code != CUSPARSE_STATUS_SUCCESS) {
        assert(code < CUSPARSE_ERROR_SIZE);
        std::cout << "GPUassert: Error " << cusparseErrors[code] << " at "
                  << file << "(" << line << ")\n";
        if (abort)
            exit(code);
    }
}