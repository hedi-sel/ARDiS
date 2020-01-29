import modulePython.dna as dna
from modulePython.read_mtx import *

import numpy as np
import random
from scipy.sparse import *
import scipy.sparse.linalg as spLnal
import time
import os
import sys

experiment = sys.argv[-1]

dampingPath = "matrixFEM/damping "+experiment+".mtx"
stiffnessPath = "matrixFEM/stiffness "+experiment+".mtx"

S = dna.ReadFromFile(stiffnessPath)
d_S = dna.D_SparseMatrix(S, True)
d_S.ConvertMatrixToCSR()
print("Stiffness matrix loaded ...")

D = dna.ReadFromFile(dampingPath)
d_D = dna.D_SparseMatrix(D, True)
d_D.ConvertMatrixToCSR()
print("Dampness matrix loaded ...")

# U = np.random.rand(d_S.cols)
pos = np.random.randint(0, d_S.cols)
length = 1
U = np.array([0.1]*pos+[10000]*length+[0.1]*(d_S.cols-length-pos))
tau = 1e-2
epsilon = 1e-6
Nit = 1000

d_U = dna.D_Array(len(U))
d_U.Fill(U)

# os.path.exists(filename)

print("Start Vector:")
print(U)

start = time.time()
M = dna.D_SparseMatrix(d_D.rows, d_D.cols)
dna.MatrixSum(d_D, d_S, -tau, M)
solve1Time = time.time() - start

start = time.time()
for i in range(0, Nit):
    d_DU = d_D.Dot(d_U)
    d_U = dna.SolveConjugateGradient(M, d_DU, epsilon)

solve2Time = time.time() - start

print("Final Vector")
d_U.Print(100000)

# C = np.array([0.5]*d_S.cols)
# d_C = dna.D_Array(len(C))
# d_C.Fill(C)
# print("Norm Difference", (d_C - d_U).Norm())


print("Run Time 1:", solve1Time)
print("Run Time 2:", solve2Time)
