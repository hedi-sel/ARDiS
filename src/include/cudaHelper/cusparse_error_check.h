#pragma once
#include <cuda_runtime.h>
#include <cusparse.h>

#define cusparseErrchk(ans)                                                    \
    { cusparseAssert((ans), __FILE__, __LINE__); }

inline void cusparseAssert(cusparseStatus_t code, const char *file, int line,
                           bool abort = true) {
    if (code != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "GPUassert: Error%i %s %d\n", code, file, line);
        if (abort)
            exit(code);
    }
}