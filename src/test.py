import modulePython.dna as dna
from modulePython.read_mtx import *

import numpy as np
from scipy.sparse import *
import scipy.sparse.linalg as spLnal
import time

# matrixPath = "matrixTest/small.mtx"
# matrixPath = "matrixTest/testNULL.mtx"

# matrixPath = "matrixSymDefPos/362_5,786_plat362.mtx"
# matrixPath = "matrixSymDefPos/1,224_56,126_bcsstk27.mtx"
# matrixPath = "matrixSymDefPos/10,605_144,579_ted_B_unscaled.mtx"
# matrixPath = "matrixSymDefPos/102,158_406,858_thermomech_TC.mtx"
matrixPath = "matrixSymDefPos/1,228,045_8,580,313_thermal2.mtx"


# Load Matrix into device memory, and convert it to compressed data type
start = time.time()
M = dna.ReadFromFile(matrixPath)
Mgpu = dna.MatrixSparse(M, True)
Mgpu.ConvertMatrixToCSR()
b = np.array([1] * M.j_size, dtype=float)

loadGpuTime = time.time() - start
start = time.time()
dna.Test(Mgpu, b)
solveGpuTime = time.time() - start
print("GPU Load time:", loadGpuTime)
print("GPU Run Time :", solveGpuTime)
