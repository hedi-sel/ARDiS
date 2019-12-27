#pragma once

#include <cusparse.h>

#include "sparseDataStruct/matrix_sparse.hpp"

void RowOrdering(MatrixSparse &d_mat);
