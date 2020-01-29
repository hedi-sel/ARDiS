#pragma once

#define USE_DOUBLE

#ifdef USE_DOUBLE
typedef double T;
#else
typedef float T;
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define NDEBUG