#pragma once

#include <cusparse.h>

#include "dataStructures/sparse_matrix.hpp"

void RowOrdering(d_spmatrix &d_mat);
