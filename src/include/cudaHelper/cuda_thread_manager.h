#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
const int BLOCK_SIZE = 32;
#endif

struct dim3Pair {
    dim3 block;
    dim3 thread;
    dim3Pair(){};
    dim3Pair(dim3 &block, dim3 &thread) : block(block), thread(thread){};
    dim3Pair(int block, int thread) : block(block), thread(thread){};
    dim3Pair(int blockX, int blockY, int blockZ, int threadX, int threadY,
             int threadZ)
        : block(blockX, blockY, blockZ), thread(threadX, threadY, threadZ){};
};

dim3Pair Make1DThreadBlock(int computeSize,
                           int blockSize = BLOCK_SIZE * BLOCK_SIZE) {
    dim3Pair threadblock(BLOCK_SIZE * BLOCK_SIZE, 1);
    if (threadblock.thread.x < computeSize) {
        threadblock.block.x = int((computeSize - 1) / threadblock.thread.x) + 1;
    } else {
        threadblock.thread.x = computeSize;
    }
    return threadblock;
}