import modulePython.dna as dna
from modulePython.read_mtx import *

import numpy as np
from scipy.sparse import *
import scipy.sparse.linalg as spLnal
import time

# matrixPath = "matrixTest/small.mtx"
# matrixPath = "matrixTest/testNULL.mtx"

# matrixPath = "matrixSymDefPos/362_5,786_plat362.mtx"
matrixPath = "matrixSymDefPos/1,224_56,126_bcsstk27.mtx"
# matrixPath = "matrixSymDefPos/10,605_144,579_ted_B_unscaled.mtx"
# matrixPath = "matrixSymDefPos/102,158_406,858_thermomech_TC.mtx"
# matrixPath = "matrixSymDefPos/1,228,045_8,580,313_thermal2.mtx"

# Load Matrix into device memory, and convert it to compressed data type
start = time.time()
M = dna.ReadFromFile(matrixPath)
Mgpu = dna.D_SparseMatrix(M, True)
Mgpu.ConvertMatrixToCSR()
loadGpuTime = time.time() - start
print("Load Gpu Time:", loadGpuTime)

A = 0
# Load Matrix on python
if(False):
    start = time.time()
    A = LoadMatrixFromFile(matrixPath, Readtype.Symetric).tocsr()
    assert (spLnal.norm(A.transpose() - A) == 0)
    loadCpuTime = time.time() - start
    print("Load CPU Time :", loadCpuTime)

print()

# Solve Linear Equation and check results
b = np.array([1] * A.shape[0], dtype=float)

# Solve GPU cholesky
if (False):
    start = time.time()
    xGpu = dna.Test(Mgpu, b)
    solveGpuTime = time.time() - start
    diff = np.linalg.norm(A.dot(xGpu) - b)
    if (diff > 0.1):
        print("Warning: Lin Equ solving gave wrong result on GPU, diff = ", diff)
    print("Solve GPU Cholesky time:", solveGpuTime)


# Solve GPU CG Method
if (True):
    start = time.time()
    xGpu = dna.Test(Mgpu, b)
    solveGpuTime = time.time() - start
    diff = np.linalg.norm(A.dot(xGpu) - b)
    if (diff > 0.1):
        print("Warning: Lin Equ solving gave wrong result on GPU, diff = ", diff)
    print("Solve GPU Cholesky time:", solveGpuTime)


# Solve CPU LU Method
if (False):
    start = time.time()
    xCpu = spLnal.spsolve(A, b)
    solveCpuTime = time.time() - start
    if (np.linalg.norm(A.dot(xCpu) - b) > 0.1):
        print("Warning: Lin Equ solving gave wrong result on CPU")
    print("Solve CPU LU time:", solveCpuTime)


print("Solve Cpu Time :", solveCpuTime)
