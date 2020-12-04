from .ardisLib import *
from .read_write import *


def to_d_spmatrix(matrix, m_type=matrix_type.COO):
    if (m_type == matrix_type.CSR):
        matrix = matrix.tocsr()
        d_matrix = d_spmatrix(
            matrix.shape[0], matrix.shape[1], matrix.indptr, matrix.indices, matrix.data, m_type)
    elif (m_type == matrix_type.COO):
        matrix = matrix.tocoo()
        d_matrix = d_spmatrix(
            matrix.shape[0], matrix.shape[1], matrix.rows, matrix.cols, matrix.data, m_type)
    elif (m_type == matrix_type.CSC):
        matrix = matrix.tocsc()
        d_matrix.d_spmatrix(
            matrix.shape[0], matrix.shape[1], matrix.indices, matrix.indptr, matrix.data, m_type)
    return d_matrix
