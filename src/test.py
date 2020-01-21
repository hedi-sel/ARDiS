import modulePython.dna as dna
from modulePython.read_mtx import *

import numpy as np
from scipy.sparse import *
import scipy.sparse.linalg as spLnal
import time

dampingPath = "matrixFEM/damping(1).mtx"
stiffnessPath = "matrixFEM/stiffness(1).mtx"

dampingPath = "matrixFEM/damping.mtx"
stiffnessPath = "matrixFEM/stiffness.mtx"

# dampingPath = "matrixTest/small.mtx"
# stiffnessPath = "matrixTest/small1.mtx"

S = dna.ReadFromFile(stiffnessPath)
d_S = dna.D_SparseMatrix(S, True)
# d_S = dna.D_SparseMatrix(S.i_size, S.j_size)

d_S.ConvertMatrixToCSR()
print("Stiffness matrix loaded ...")

D = dna.ReadFromFile(dampingPath)
d_D = dna.D_SparseMatrix(D, True)
d_D.ConvertMatrixToCSR()
print("Dampness matrix loaded ...")


U = np.random.rand(d_S.j_size)
tau = 0.01
epsilon = 1e-2

d_U = dna.D_Array(len(U))
d_U.Fill(U)

###
# M = D-tau*S
###
start = time.time()
M = dna.D_SparseMatrix()
dna.MatrixSum(d_D, d_S, -tau, M)
d_DU = dna.D_Array(len(U))
solve1Time = time.time() - start

start = time.time()
V1 = dna.SolveConjugateGradient(M, d_D.Dot(d_U), epsilon)
solve2Time = time.time() - start


print("Run Time 1:", solve1Time)
print("Run Time 2:", solve2Time)
