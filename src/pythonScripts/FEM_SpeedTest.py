import modulePython.dna as dna
from modulePython.read_mtx import *

from scipy.sparse import *
import scipy.sparse.linalg as spLnal
import time
import os

from modulePython._parameter import *

S = dna.ReadFromFile(stiffnessPath)
d_S = dna.D_SparseMatrix(S, True)
d_S.ConvertMatrixToCSR()
print("Stiffness matrix loaded ...")

D = dna.ReadFromFile(dampingPath)
d_D = dna.D_SparseMatrix(D, True)
d_D.ConvertMatrixToCSR()
print("Dampness matrix loaded ...")

d_U = dna.D_Array(len(U))
d_U.Fill(U)

d_Ones = dna.D_Array(len(U))
d_Ones.Fill(np.ones(len(U)))

start = time.time()
d_S *= tau
M = d_D + d_S
solve1Time = time.time() - start

start = time.time()
for i in range(0, Nit):
    d_DU = d_D.Dot(d_U)
    dna.SolveConjugateGradientRawData(M, d_DU, d_U, epsilon)
    # d_U.Print(30)
    # print(d_U.Dot(d_Ones))

solve2Time = time.time() - start

print("Final Vector")
d_U.Print()

# C = np.array([0.5]*d_S.cols)
# d_C = dna.D_Array(len(C))
# d_C.Fill(C)
# print("Norm Difference", (d_C - d_U).Norm())


print("Run Time 1:", solve1Time)
print("Run Time 2:", solve2Time)
