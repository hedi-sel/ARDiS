#pragma once

#include "dataStructures/array.hpp"

class D_Mesh {
  public:
    D_Vector X;
    D_Vector Y;

    __host__ __device__ int size();

    D_Mesh(D_Vector &X, D_Vector &Y);
    ~D_Mesh();
};