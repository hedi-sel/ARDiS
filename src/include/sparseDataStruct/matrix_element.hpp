#pragma once

#include "constants.h"

class MatrixElement
{
public:
    const int i;
    const int j;
    const T val;
    MatrixElement(int i, int j, const T &val) : i(i), j(j), val(val){};
    ~MatrixElement(){};

    void Print() const
    {
        printf("%i, %i: %f\n", i, j, val);
    }
};