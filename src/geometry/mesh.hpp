#pragma once

#include "dataStructures/array.hpp"

class d_mesh {
  public:
    d_vector X;
    d_vector Y;

    __host__ __device__ int size();

    d_mesh(int n);
    d_mesh(int n, T *x, T *y); // Initialize from a host pointer
    d_mesh(d_vector &X, d_vector &Y);
    ~d_mesh();
};