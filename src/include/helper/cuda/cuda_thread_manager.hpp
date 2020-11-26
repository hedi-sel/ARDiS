#pragma once

#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
const int BLOCK_SIZE = 32;
#endif

struct dim3Pair {
    dim3 block;
    dim3 thread;
    dim3Pair();
    dim3Pair(dim3 &block, dim3 &thread);
    dim3Pair(int block, int thread);
    dim3Pair(int blockX, int blockY, int threadX, int threadY);
    dim3Pair(int blockX, int blockY, int blockZ, int threadX, int threadY,
             int threadZ);
};

dim3Pair Make1DThreadBlock(int computeSize,
                           int blockSize = BLOCK_SIZE * BLOCK_SIZE);

dim3Pair MakeWide1DThreadBlock(int computeSize, int width,
                               int blockSize = BLOCK_SIZE * BLOCK_SIZE);