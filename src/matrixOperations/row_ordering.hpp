#pragma once

#include <cusparse.h>

#include "dataStructures/sparse_matrix.hpp"

void RowOrdering(D_SparseMatrix &d_mat);
