from .ardisLib import *
from .read_mtx import *


def ToD_SparseMatrix(matrix, m_type=MatrixType.COO):
    if (m_type == MatrixType.CSR):
        matrix = matrix.tocsr()
        d_matrix = D_SparseMatrix(
            matrix.shape[0], matrix.shape[1], matrix.indptr, matrix.indices, matrix.data, m_type)
    elif (m_type == MatrixType.COO):
        matrix = matrix.tocoo()
        d_matrix = D_SparseMatrix(
            matrix.shape[0], matrix.shape[1], matrix.rows, matrix.cols, matrix.data, m_type)
    elif (m_type == MatrixType.CSC):
        matrix = matrix.tocsc()
        d_matrix.D_SparseMatrix(
            matrix.shape[0], matrix.shape[1], matrix.indices, matrix.indptr, matrix.data, m_type)
    return d_matrix
