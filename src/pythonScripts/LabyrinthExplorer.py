import modulePython.dna as dna
from modulePython.read_mtx import *
import modulePython.reaction_diffusion as rd
from modulePython.concentration_manager import *

from scipy.sparse import *
import scipy.sparse.linalg as spLnal
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as colors
from matplotlib import ticker, cm
import numpy as np
import time
import math
import os

matrixFolder = "matrixLabyrinth"
printFolder = "output"

tau = 1e-2
epsilon = 1e-2
Tmax = 4

plot_dt = 0.1

use_system = False

def ExploreLabyrinth(out, ins, reaction=5, output="None", return_item="Success", verbose=True):
    if True:  # Prepare output type and return item
        DoPlot = False
        DoRecordResult = False
        if (output == "Plot"):
            DoPlot = True
        elif (output == "RecordResult"):
            DoRecordResult = True
        elif (output != "None"):
            print("Error! Output type not recognized")
            return
        if return_item != "Success" and return_item != "Time" and return_item != "Dispersion":
            print("Error! Return item not recognized")
            return

    if True:  # Make Mesh and Matrices. Load them from file
        name = str(round(out, 2))+"_"+str(round(ins, 2)) + "_"+str(round(reaction, 2))
        os.system("./MakeMatrix.wls "+name+" "+str(out)+" "+str(ins))
        dampingPath = matrixFolder+"/damping "+name+".mtx"
        stiffnessPath = matrixFolder+"/stiffness "+name+".mtx"

        d_S = rd.ToD_SparseMatrix(LoadMatrixFromFile(stiffnessPath, Readtype.Symetric), dna.MatrixType.CSR)
        if(verbose):
            print("Stiffness matrix loaded ...")
        d_D = rd.ToD_SparseMatrix(LoadMatrixFromFile(dampingPath, Readtype.Symetric), dna.MatrixType.CSR)
        if (verbose):
            print("Dampness matrix loaded ...")

        Mesh = LoadMeshFromFile(matrixFolder+"/mesh "+name+".dat")

    if True:  # Set initial vector and prepare FEM resolution
        n = d_S.cols

        pos = np.random.randint(0, n)
        pos = 0
        length = 1
        fillVal = 0.001
        U = np.array([fillVal] * n)
        FillZone(U, Mesh, RectangleZone(-1, 2, -1, -0.8), 1)
        d_U = dna.D_Array(len(U))
        d_U.Fill(U)
        # Mesh_X = dna.D_Array(len(U))
        # Mesh_Y = dna.D_Array(len(U))
        # Mesh_X.Fill(Mesh.x)
        # Mesh_Y.Fill(Mesh.y)

        # dna.FillZone(d_U, Mesh_X, Mesh_Y,
        #              dna.RectangleZone(-1, 2, -1, -0.8), 1)

        print("Start state:")
        if(use_system):
            system.Print()
        else:
            d_U.Print()

        start = time.time()
        d_S *= tau
        d_M = d_D + d_S
        solve1Time = time.time() - start

    plotPeriod = int(plot_dt/tau)
    os.system("rm -rf "+printFolder+"/"+name)
    os.system("mkdir "+printFolder+"/"+name+" 2> ./null")
    start = time.time()
    k = 0
    Nit = int(Tmax / tau)
    ZoneArrivee = RectangleZone(-1, -0.9, 0, 2)
    success = False
    FirstFinish = 0
    LastFinish = 0
    MeanFinish = 0
    threashold = 0.5

    for i in range(0, Nit):
        if use_system:
            system.IterateDiffusion(tau)
            system.Prune(fillVal)
            system.IterateReaction(tau)
            U = system.GetSpecies("U").ToNumpyArray()
        else:
            d_DU = d_D.Dot(d_U)
            dna.SolveConjugateGradientRawData(d_M, d_DU, d_U, epsilon)
            U = d_U.ToNumpyArray()

        if DoPlot:  # Ploting
            if plotPeriod != 0 and i % plotPeriod == 0:
                fig, ax = plt.subplots()
                ax.set_aspect('equal')
                tcf = ax.tricontourf(Mesh,  np.abs(U), np.logspace(math.log10(
                    1.e-5), math.log10(1), 11), norm=colors.LogNorm(vmin=1.e-5, vmax=1), extend='neither')
                fig.colorbar(tcf, extend='max')
                title = str(i//plotPeriod)
                ax.set_title(title+"s")
                plt.savefig(printFolder+"/"+name+"/"+title+".png")
                plt.close()
        if verbose:
            if Nit >= 100 and i >= k * Nit / 10 and i < k * Nit / 10 + 1:
                print(str(k * 10) + "% completed")
                k += 1

        if MeanFinish == 0 and (GetMeanZone(U, Mesh, ZoneArrivee) > np.mean(U)*threashold):
            MeanFinish = i * tau
        if FirstFinish == 0 and (GetMaxZone(U, Mesh, ZoneArrivee) > np.mean(U)*threashold):
            FirstFinish = i*tau
        if LastFinish == 0 and (GetMinZone(U, Mesh, ZoneArrivee) > np.mean(U)*threashold):
            success = True
            LastFinish = i*tau
            break

    Dispersion = (LastFinish-FirstFinish)

    if DoPlot:
        os.system("convert -delay 10 -loop 0 $(ls -1 "+printFolder+"/" +
                name+"/*png | sort -V) "+printFolder+"/"+name+".gif")
        os.system("rm -rf "+printFolder+"/"+name)

    solve2Time = time.time() - start

    if(verbose):
        if (success):
            print("Labyrinthe traversé! Arrivée à t= ", MeanFinish, "s")
            print("Dispersion du front : ", Dispersion, "s")
        else:
            print("Labyrinth not completed")
            print("Final State")
            if(use_system):
                system.Print()
            else:
                d_U.Print()            
            print("Run Time:", solve2Time)

    if DoRecordResult:
        os.system("echo " + name + " " + str(round(MeanFinish, 2)) +
                  " " + str(round(Dispersion, 4)) + " >> output/results.out")

    if(return_item == "Success"):
        return success
    elif (return_item == "Time"):
        return MeanFinish
    elif (return_item == "Dispersion"):
        return Dispersion

