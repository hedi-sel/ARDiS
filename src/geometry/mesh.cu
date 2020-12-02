#include "mesh.hpp"
#include <cuda_runtime.h>

d_mesh::d_mesh(int n) : X(n), Y(n) {}
d_mesh::d_mesh(int n, T *x, T *y) : X(n), Y(n) {}
d_mesh::d_mesh(d_vector &X, d_vector &Y) : X(X), Y(Y) { assert(X.n == Y.n); }

__host__ __device__ int d_mesh::size() { return X.n; }

d_mesh::~d_mesh() {}