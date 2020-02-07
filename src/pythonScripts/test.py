import modulePython.dna as dna
from modulePython.read_mtx import *

import numpy as np
from scipy.sparse import *
import scipy.sparse.linalg as spLnal
import time

experiment = "1"
dampingPath = "matrixFEM/damping "+experiment+".mtx"
stiffnessPath = "matrixFEM/stiffness " + experiment + ".mtx"

start = time.time()

S = ReadMatrixFromFile(stiffnessPath)
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
pos = 0
U = np.array([0.0]*pos+[1]*length+[0.0]*(d_S.cols-length-pos))

tau = 1e-3
epsilon = 1e-3
Nit = 1000

system = dna.System(len(U))
# Coeff diffusion; Stabilité?==Disponibilité==Pairabilité (k- k+); Affinité avec l'exonucléase
system.AddSpecies("U")
# Enelever une espece
system.SetSpecies("U", U)

print("Start Vector:")
system.Print()

system.LoadStiffnessMatrix(d_S)
system.LoadDampnessMatrix(d_D)

system.AddReaction("U", 1, "U", 2, 100)
system.AddReaction("U", 2, "U", 1, 100)

solve1Time = time.time() - start

start = time.time()
for i in range(0, Nit):
    system.IterateDiffusion(tau)
    system.Prune()
    system.IterateReaction(tau)
    system.State.Print()

solve2Time = time.time() - start

print("Final Vector")
system.Print()

# C = np.array([0.5]*d_S.cols)
# d_C = dna.D_Array(len(C))
# d_C.Fill(C)
# print("Norm Difference", (d_C - d_U).Norm())


print("Run Time 1:", solve1Time)
print("Run Time 2:", solve2Time)
