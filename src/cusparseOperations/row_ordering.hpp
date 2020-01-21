#pragma once

#include <cusparse.h>

#include "sparseDataStruct/matrix_sparse.hpp"

void RowOrdering(D_SparseMatrix &d_mat);
