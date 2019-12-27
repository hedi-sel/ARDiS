import modulePython.dna as dna
import modulePython.read_mtx as read_mtx

import numpy as np
from scipy.sparse import *
import scipy.sparse.linalg as spLnal
import time

# matrixPath = "matrix/small.mtx"
matrixPath = "matrix/494_bus.mtx"
# matrixPath = "matrix/1138_bus.mtx"
# matrixPath = "matrix/NonLinearDiff9801.mtx"
# matrixPath = "matrix/Baumann112211.mtx"

# Load Matrix into device memory, and convert it to compressed data type
start = time.time()
M = dna.ReadFromFile(matrixPath)
Mgpu = dna.MatrixSparse(M, True)
Mgpu.ConvertMatrixToCSR()
loadGpuTime = time.time() - start

# Load Matrix on python
start = time.time()
A = read_mtx.LoadMatrixFromFile(matrixPath).tocsr()
loadCpuTime = time.time() - start

# Solve Linear Equation and check results
b = np.array([1] * A.shape[0], dtype=float)

start = time.time()
xGpu = dna.SolveLinEq(Mgpu, b)
solveGpuTime = time.time() - start
print((A.dot(xGpu)))

start = time.time()
xCpu = spLnal.spsolve(A, b)
solveCpuTime = time.time() - start
# print(A.dot(xCpu))

print("Load Gpu time:", loadGpuTime)
print("Solve Gpu Time :", solveGpuTime)
print()
print("Load Cpu time:", loadCpuTime)
print("Solve Cpu Time :", solveCpuTime)
