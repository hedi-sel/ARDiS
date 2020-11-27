#include "mesh.hpp"
#include <cuda_runtime.h>

D_Mesh::D_Mesh(D_Vector &X, D_Vector &Y) : X(X), Y(Y) { assert(X.n == Y.n); }

__host__ __device__ int D_Mesh::size() { return X.n; }

D_Mesh::~D_Mesh() {}