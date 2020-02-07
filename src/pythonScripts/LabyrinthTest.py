import modulePython.dna as dna
from modulePython.read_mtx import *

from scipy.sparse import *
import scipy.sparse.linalg as spLnal
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import time
import os

matrixFolder = "matrixLabyrinth"
printFolder = "output"

tau = 1e-3
epsilon = 1e-3
Tmax = 10

plot_dt = 0.1


def MakeLabyrinth(a, b):

    if True: #Make Mesh and Matrices. Load them from file
        name = str(min( a,b))+"_"+str(max(a,b))
        os.system("./MakeMatrix.wls "+name+" "+str(a)+" "+str(b))
        dampingPath = matrixFolder+"/damping "+name+".mtx"
        stiffnessPath = matrixFolder+"/stiffness "+name+".mtx"

        S = dna.ReadFromFile(stiffnessPath)
        d_S = dna.D_SparseMatrix(S, True)
        d_S.ConvertMatrixToCSR()
        print("Stiffness matrix loaded ...")

        D = dna.ReadFromFile(dampingPath)
        d_D = dna.D_SparseMatrix(D, True)
        d_D.ConvertMatrixToCSR()
        print("Dampness matrix loaded ...")

        Mesh = LoadMeshFromFile(matrixFolder+"/mesh "+name+".dat")

    if True: #Set initial vector and prepare FEM resolution
        n = d_S.cols

        pos = np.random.randint(0, n)
        pos = 0
        length = 1
        fillVal = 0.0001
        U = np.array([fillVal]*pos+[1000]*length+[fillVal]*(n-length-pos))

        print("Start Vector:")
        print(U)

        d_U = dna.D_Array(len(U))
        d_U.Fill(U)
        
        start = time.time()
        d_S *= tau
        M = d_D + d_S
        solve1Time = time.time() - start

    plotPeriod = int(plot_dt/tau)
    os.system("mkdir "+printFolder+"/"+name+" 2> ./null")
    start = time.time()
    k = 0
    Nit = int(Tmax / tau)
    for i in range(0, Nit):
        d_DU = d_D.Dot(d_U)
        dna.SolveConjugateGradientRawData(M, d_DU, d_U, epsilon)

        if True: #Ploting code
            if plotPeriod != 0 and i % plotPeriod == 0:
                fig, ax = plt.subplots()
                ax.set_aspect('equal')
                tcf = ax.tricontourf(Mesh, d_U.ToNumpyArray(),np.exp(np.linspace(np.ln(0.000001),np.ln(0.01),10)))
                fig.colorbar(tcf, boundaries = np.linspace(0,1,100))
                title = str(i//plotPeriod)
                ax.set_title(title+"s")
                plt.savefig(printFolder+"/"+name+"/"+title+".png")
                plt.close()
            if Nit >= 100 and i >= k * Nit / 10 and i < k * Nit / 10 + 1:
                print(str(k * 10) + "% completed")
                k += 1

    os.system("convert -delay 10 -loop 0 $(ls -1 "+printFolder+"/" +
              name+"/*png | sort -V) "+printFolder+"/"+name+".gif")
    os.system("rm -r "+printFolder+"/"+name)
    
    solve2Time = time.time() - start

    print("Final Vector")
    d_U.Print()

    print("Run Time 1:", solve1Time)
    print("Run Time 2:", solve2Time)


MakeLabyrinth(2, 1)
