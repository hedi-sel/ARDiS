import modulePython.dna as dna
from modulePython.read_mtx import *
from modulePython.cg_solve import *
from modulePython.concentration_manager import *
from LabyrinthExplorer import *

from scipy.sparse import *
import scipy.sparse.linalg as spLnal
import time
import os

matrixFolder = "matrixLabyrinth"

dt = 1.e-2
Nit = 5000
epsilon = 1.e-3
drain = 1.e-8
reaction = 1

stepSizes = ["100", "50", "30", "20", "15", "10", "5", "3", "2", "1", "0.5", "0.2"]

dataName = "precision="+stepSizes[11]

diffusion = 0
diffusion_init = 0
reaction = 0


start = time.time()

if True:
    dampingPath = matrixFolder+"/damping_"+dataName+".mtx"
    stiffnessPath = matrixFolder+"/stiffness_"+dataName+".mtx"
    meshPath = matrixFolder + "/mesh_" + dataName + ".dat"

    # os.system("./ImageToMatrix.wls "+dataName)
    S = LoadMatrixFromFile(stiffnessPath, Readtype.Symetric)
    S.tocsr()
    print("Stiffness matrix loaded ...")

    D = LoadMatrixFromFile(dampingPath, Readtype.Symetric)
    D.tocsr()
    print("Dampness matrix loaded ...")


    Mesh = LoadMeshFromFile(meshPath)

    n = len(Mesh.x)

    fillVal = 0.0
    U = np.full(n, fillVal)
    StartZone = RectangleZone(0, 0, 500, 10)
    FillZone(U, Mesh, StartZone, 1.0)

    loading_time = time.time() - start

    print("loading_time:", loading_time)

    start = time.time()

    start2 = time.time()
    M = D - dt * S
    diffusion_init += time.time() - start2

    N = U.copy()
    P = U.copy()

name = "cppTest"
os.system("rm -f "+csvFolder+"/"+name+".csv")

Z = np.zeros(len(N))
for i in range(0, Nit):
    time2 = time.time()
    CGNaiveSolve(M, D.dot(N), N, epsilon)
    last_nIter = CGNaiveSolve(M, D.dot(P), P, epsilon)
    diffusion += time.time() - time2
        
    os.system("echo "+str(i*dt)+"'\t'"+str(last_nIter)+" >> output/CgmIterCount")

    time2 = time.time()
    N -= drain
    P-= drain
    N=np.maximum(N, Z)
    P=np.maximum(P, Z)
    N += reaction * N * dt
    # P+= reaction* P* dt
    prog = reaction*dt*P*N
    P += prog
    N -= prog
    reaction += time.time() - time2

    # dna.ToCSV(N, csvFolder+"/"+name+".csv" )

    # if (i > 0 and np.linalg.norm(N - oldN) < 1.e-3):
    #     print("Exploration Suceeded in",i*dt,"s")
    #     break
    # oldN = N.copy()

solve2Time = time.time() - start
print("Run Time:", solve2Time)
print("loading_time:", loading_time)


print("diffusion\n", diffusion)
print("diffusion_init\n", diffusion_init)
print("reaction\n", reaction)
PrintProf()


# PrintLabyrinth(name, verbose=True, plotEvery=100, dt=dt, meshPath =meshPath)

# print("Final Vector")
# print(U)

