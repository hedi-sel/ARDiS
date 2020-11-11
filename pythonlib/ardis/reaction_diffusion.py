from .ardisLib import *
from .read_mtx import *


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


class State:
    n = 0

    names = {}
    data = []

    def __init__(self, n):
        self.n = n

    def AddSpecies(self, name):
        self.data.append(D_Vector(self.n))
        self.names[name] = len(self.data)-1

    def GetSpecies(self, name):
        return self.data[self.names[name]]

    def SetSpecies(self, name, array):
        self.data[self.names[name]].Fill(array)

    def Print(self, i=5):
        for name in self.names:
            print(name, " :")
            self.data[self.names[name]].Print()


# class System(State):
#     d_D = 0
#     d_S = 0
#     d_M = 0
#     last_dt = 0
#     epsilon = 1.e-3

#     Reactions = []

#     def LoadStiffnessMatrix(self, d_S):
#         self.d_S = d_S

#     def LoadDampnessMatrix(self, d_D):
#         self.d_D = d_D

#     def IterateDiffusion(self, dt):
#         if self.last_dt != dt:
#             if type(self.d_D) == int or type(self.d_S) == int:
#                 print("Error: you have to load damping and stiffness matrices first")
#             else:
#                 self.d_M = self.d_D + self.d_S * dt
#                 self.last_dt = dt

#         for d_U in self.data:
#             d_DU = self.d_D.Dot(d_U)
#             SolveConjugateGradientRawData(self.d_M, d_DU, d_U, self.epsilon)

#     def AddReaction(self, reag, kr, prod, kp, rate=1):
#         Reactions = Reactions + [([(reag, kr)], [(prod, kp)], rate)]

#     def IterateReaction(self, dt):
#         print("No reaction")
