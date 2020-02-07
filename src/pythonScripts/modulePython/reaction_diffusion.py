from modulePython.dna import *
from modulePython.read_mtx import *


def ToD_SparseMatrix(matrix, m_type=MatrixType.COO):
    d_matrix = D_SparseMatrix(
        matrix.shape[0], matrix.shape[1], matrix.nnz, m_type)
    if (m_type == MatrixType.CSR):
        matrix = matrix.tocsr()
        d_matrix.Fill(matrix.indptr, matrix.indices, matrix.data)
    elif (m_type == MatrixType.COO):
        matrix = matrix.tocoo()
        d_matrix.Fill(matrix.rows, matrix.cols, matrix.data)
    elif (m_type == MatrixType.CSC):
        matrix = matrix.tocsc()
        d_matrix.Fill(matrix.indices, matrix.indptr, matrix.data)
    return d_matrix
