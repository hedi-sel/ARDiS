import modulePython.dna as dna
from modulePython.read_mtx import *

import numpy as np
import random
from scipy.sparse import *
import scipy.sparse.linalg as spLnal
import time
import os


experiment = sys.argv[-1]
if experiment == "":
    experiment = input("Which matrix?")

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
tau = 1e-5
epsilon = 1e-6
Nit = 1000

d_U = dna.D_Array(len(U))
d_U.Fill(U)

file_path = '../output/' + experiment+".dat"
if (os.path.exists(file_path)):
    os.remove(file_path)
# os.system("touch "+file_path)
outputFile = open('output/' + experiment+".dat", 'w')


def write(array):
    for i in range(0, U.size):
        outputFile.write(""+str(array[i]) + " ")
    outputFile.write("\n")


print("Preparation ready, starting iterations")

start = time.time()
M = dna.D_SparseMatrix(d_D.rows, d_D.cols)
dna.MatrixSum(d_D, d_S, -tau, M)
solve1Time = time.time() - start

write(d_U.ToNumpyArray())

start = time.time()
for i in range(0, Nit):
    d_DU = d_D.Dot(d_U)
    d_U = dna.SolveConjugateGradient(M, d_DU, epsilon)
    write(d_U.ToNumpyArray())

solve2Time = time.time() - start

print("Job Done!")
