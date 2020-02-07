import modulePython.dna as dna
from modulePython.read_mtx import *
from modulePython.cg_solve import *

from scipy.sparse import *
import scipy.sparse.linalg as spLnal
import time
import os

from modulePython._parameter import *

S = LoadMatrixFromFile(stiffnessPath, Readtype.Symetric)
S.tocsr()
print("Stiffness matrix loaded ...")

D = LoadMatrixFromFile(dampingPath, Readtype.Symetric)
D.tocsr()
print("Dampness matrix loaded ...")

M = D - tau * S

print(M.shape, U.shape)
start = time.time()
for i in range(0, Nit):
    CGNaiveSolve(M, D.dot(U), U, epsilon)
    # print(U)
    # print(U.dot(np.ones(len(U))))

solve2Time = time.time() - start

print("Final Vector")
print(U)

print("Run Time 2:", solve2Time)
