#include "cuda_thread_manager.hpp"

dim3Pair::dim3Pair(){};
dim3Pair::dim3Pair(dim3 &block, dim3 &thread) : block(block), thread(thread){};
dim3Pair::dim3Pair(int block, int thread) : block(block), thread(thread){};
dim3Pair::dim3Pair(int blockX, int blockY, int threadX, int threadY)
    : block(blockX, blockY), thread(threadX, threadY){};
dim3Pair::dim3Pair(int blockX, int blockY, int blockZ, int threadX, int threadY,
                   int threadZ)
    : block(blockX, blockY, blockZ), thread(threadX, threadY, threadZ){};

dim3Pair make1DThreadBlock(int computeSize, int blockSize) {
    return MakeWide1DThreadBlock(computeSize, 1, blockSize);
}

dim3Pair MakeWide1DThreadBlock(int computeSize, int width, int blockSize) {
    int exp = 0;
    while ((1 << exp) < width)
        exp++;
    dim3Pair threadblock(1, 1, blockSize / (1 << exp), (1 << exp));
    if (threadblock.thread.x < computeSize) {
        threadblock.block.x = int((computeSize - 1) / threadblock.thread.x) + 1;
    } else {
        threadblock.thread.x = computeSize;
    }
    return threadblock;
}