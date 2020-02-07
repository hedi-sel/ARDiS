import modulePython.dna as dna
from modulePython.read_mtx import *

import numpy as np
import random
from scipy.sparse import *
import scipy.sparse.linalg as spLnal
import time
import os
import sys

S = dna.ReadFromFile(stiffnessPath)
d_S = dna.D_SparseMatrix(S, True)
d_S.ConvertMatrixToCSR()
print("Stiffness matrix loaded ...")

D = dna.ReadFromFile(dampingPath)
d_D = dna.D_SparseMatrix(D, True)
d_D.ConvertMatrixToCSR()
print("Dampness matrix loaded ...")

system = dna.System(len(U))
system.AddSpecies("U")
system.SetSpecies("U", U)
system.IterateDiffusion(tau)
system.LoadStiffnessMatrix(d_S)
system.LoadDampnessMatrix(d_D)

system.AddReaction("U", 1, "U", 2, 10)
system.AddReaction("U", 2, "U", 1, 1)


file_path = '../output/' + experiment+".dat"
if (os.path.exists(file_path)):
    os.remove(file_path)
outputFile = open('output/' + experiment+".dat", 'w')


def write(array):
    for i in range(0, U.size):
        outputFile.write(""+str(array[i]) + " ")
    outputFile.write("\n")


print("Preparation ready, starting iterations")


solve1Time = time.time() - start

start = time.time()

write(system.State.GetSpecies("U").ToNumpyArray())

plotPeriod = int(plot_dt/dt)
k = 0
for i in range(0, Nit):
    system.IterateDiffusion(tau)
    system.Prune()
    system.IterateReaction(tau)
    system.Prune()
    if i % plotPeriod == 0:
        write(system.State.GetSpecies("U").ToNumpyArray())
    if Nit >= 100:
        if i >= k * Nit / 10 and i < k * Nit / 10 + 1:
            print(str(k * 10) + "% completed")
            k += 1

solve2Time = time.time() - start

print("Job Done!")

print("Run Time 1:", solve1Time)
print("Run Time 2:", solve2Time)
