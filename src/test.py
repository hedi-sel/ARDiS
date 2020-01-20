import modulePython.dna as dna
from modulePython.read_mtx import *

import numpy as np
from scipy.sparse import *
import scipy.sparse.linalg as spLnal
import time

dampingPath = "matrixTest/damping.mtx"
stiffnessPath = "matrixTest/stiffness.mtx"

dampingPath = "matrixTest/small.mtx"
stiffnessPath = "matrixTest/small.mtx"

D = dna.ReadFromFile(dampingPath)
d_D = dna.MatrixSparse(D, True)
d_D.ConvertMatrixToCSR()
print("Dampness matrix loaded ...")

S = dna.ReadFromFile(stiffnessPath)
d_S = dna.MatrixSparse(S, True)
d_S.ConvertMatrixToCSR()
print("Stiffness matrix loaded ...")

u = np.array([1] * d_S.j_size, dtype=float)

A = LoadMatrixFromFile(stiffnessPath, Readtype.Symetric).tocsr()
print("Test matrix loaded on CPU ...")

start = time.time()
V1 = dna.Test(d_S, d_D, u)
V2 = A.dot(u)

print("Norm diff = ", np.linalg.norm(V1 - V2))

solveGpuTime = time.time() - start


print("GPU Run Time :", solveGpuTime)
