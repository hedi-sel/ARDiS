import argparse
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
from enum import Enum

matrixFolder = "matrixLabyrinth"
printFolder = "output"


class OutputType(Enum):
    NONE = 0
    PLOT = 1
    RECORD = 2


class ReturnType(Enum):
    SUCCESS = 0
    TIME = 1
    DISPERSION = 2


def ExploreLabyrinth(out, ins, thickness=1, reaction=5, output=OutputType.NONE, return_item=ReturnType.SUCCESS, verbose=True, dt=1e-2, epsilon=1e-3, max_time=3, plot_dt=0.1, guess=float(0), use_system=True,
                     threashold=0.9):
    if True:  # Prepare output type and return item
        DoPlot = False
        DoRecordResult = False
        if (output == OutputType.PLOT):
            DoPlot = True
        elif (output == OutputType.RECORD):
            DoRecordResult = True
        elif (output != OutputType.NONE):
            print("Error! Output type not recognized")
            return
        if return_item != ReturnType.SUCCESS and return_item != ReturnType.TIME and return_item != ReturnType.DISPERSION:
            print("Error! Return item not recognized")
            return

    if True:  # Make Mesh and Matrices. Load them from file
        name = str(round(out, 2))+"_"+str(round(ins, 2)) + \
            "_"+str(round(thickness, 2))+"_"+str(round(reaction, 2))
        os.system("./MakeMatrix.wls "+name+" "+str(out) +
                  " "+str(ins)+" "+str(thickness))
        dampingPath = matrixFolder+"/damping_"+name+".mtx"
        stiffnessPath = matrixFolder+"/stiffness_"+name+".mtx"
        meshPath = matrixFolder + "/mesh_" + name + ".dat"

        d_S = rd.ToD_SparseMatrix(LoadMatrixFromFile(
            stiffnessPath, Readtype.Symetric), dna.MatrixType.CSR)
        if(verbose):
            print("Stiffness matrix loaded ...")
        d_D = rd.ToD_SparseMatrix(LoadMatrixFromFile(
            dampingPath, Readtype.Symetric), dna.MatrixType.CSR)
        if (verbose):
            print("Dampness matrix loaded ...")

        Mesh = LoadMeshFromFile(meshPath)

    if True:  # Set initial vector and prepare FEM resolution
        n = d_S.cols

        fillVal = 0.001
        U = np.full(n, fillVal)
        StartZone = RectangleZone(0, 0, 10, 0.2)
        FillZone(U, Mesh, StartZone, 1)

        if (use_system):
            system = dna.System(d_D.cols)
            system.AddSpecies("U")
            system.SetSpecies("U", U)

            system.LoadStiffnessMatrix(d_S)
            system.LoadDampnessMatrix(d_D)

            system.AddReaction("U", 1, "U", 2, reaction)
            system.AddReaction("U", 2, "U", 1, reaction)
        else:
            d_U = dna.D_Array(len(U))
            d_U.Fill(U)

            d_S *= dt
            d_M = d_D + d_S

        if verbose:
            print("Start state:")
            if(use_system):
                system.Print()
            else:
                d_U.Print()

    start = time.time()

    if (DoPlot):
        plotPeriod = int(plot_dt/dt)
        os.system("rm -rf "+printFolder+"/"+name)
        os.system("mkdir " + printFolder + "/" + name + " 2> ./null")

    start = time.time()
    k = 0
    Nit = int(max_time / dt)

    nTests = 10
    angles = np.linspace(0, math.pi/2, nTests+1)
    TestZones = []
    for i in range(0, nTests):
        TestZones = TestZones + [TriangleZone(0, 0, 3*math.cos(angles[i]), 3*math.sin(
            angles[i]), 3 * math.cos(angles[i + 1]), 3 * math.sin(angles[i + 1]))]
    TestMinTime = [0]*nTests
    TestMaxTime = [0]*nTests
    currentMinTest = 0
    currentMaxTest = 0

    FinishZone = RectangleZone(0, 0, 0.2, 10)
    Finished = False
    FinishTime = 0

    for i in range(0, Nit):
        if use_system:
            system.IterateDiffusion(dt)
            # d_DU = d_D.Dot(system.GetSpecies("U"))
            # dna.SolveConjugateGradientRawData(
            #     d_M, d_DU, system.GetSpecies("U"), epsilon)
            #
            # system.Prune(fillVal)
            system.IterateReaction(dt)
            U = system.GetSpecies("U").ToNumpyArray()
        else:
            d_DU = d_D.Dot(d_U)
            dna.SolveConjugateGradientRawData(d_M, d_DU, d_U, epsilon)
            U = d_U.ToNumpyArray()

        if DoPlot:  # Ploting
            if plotPeriod != 0 and i % plotPeriod == 0:
                fig, ax = plt.subplots()
                ax.set_aspect('equal')
                # tcf = ax.tricontourf(Mesh,  np.abs(U), np.logspace(math.log10(
                #     1.e-5), math.log10(1), 11), norm=colors.LogNorm(vmin=1.e-5, vmax=1), extend='neither')
                # fig.colorbar(tcf, extend='max')
                title = str(i//plotPeriod)
                # ax.set_title(title)
                plt.scatter(Mesh.x, Mesh.y, c=U, alpha=0.3, vmin=0, vmax=1)
                plt.savefig(printFolder+"/"+name+"/"+title+".png")
                plt.close()
        if verbose:
            if Nit >= 100 and i >= k * Nit / 10 and i < k * Nit / 10 + 1:
                print(str(k * 10) + "% completed")
                k += 1
        if (i * dt >= guess):  # Start checking if finished
            while currentMinTest < nTests and (GetMinZone(U, Mesh, TestZones[currentMinTest]) > threashold):
                TestMinTime[currentMinTest] = i * dt
                currentMinTest += 1
            while currentMaxTest < nTests and (GetMaxZone(U, Mesh, TestZones[currentMaxTest]) > threashold):
                TestMaxTime[currentMaxTest] = i * dt
                currentMaxTest += 1

            if (GetMinZone(U, Mesh, FinishZone) > threashold):  # currentTest == nTests and
                Finished = True
                FinishTime = i*dt
                break
            if (i - 1) * dt < guess and currentMaxTest > 0:
                if guess == 0:
                    print("Error, reached the test in one iteration or less!")
                    return
                if verbose:
                    print(
                        "Reached a test zone before the guess, we will start over without using the guess!")
                return ExploreLabyrinth(out, ins, reaction=reaction, output=output, return_item=return_item, verbose=verbose, dt=dt, epsilon=epsilon, max_time=max_time, plot_dt=plot_dt, guess=0.0)

    # Dispersion = (LastFinish-FirstFinish)

    if DoPlot:
        os.system("convert -delay 10 -loop 0 $(ls -1 "+printFolder +
                  "/" + name+"/*png | sort -V) "+printFolder+"/"+name+".gif")
        os.system("rm -rf "+printFolder+"/"+name)

    solve2Time = time.time() - start

    if(verbose):
        if (Finished):
            print("Labyrinthe traversé! Arrivée à t= ", FinishTime, "s")
            print("Exterior times ", TestMinTime)
            print("Interior times ", TestMaxTime)
            # print("Dispersion du front : ", Dispersion, "s")
        else:
            print("Labyrinth not completed")
            print("Final State")
            if(use_system):
                system.Print()
            else:
                d_U.Print()
        print("Run Time:", solve2Time)
        if (DoPlot):
            print("Results plot have been saved here: " +
                  printFolder+"/"+name+".gif")

    if DoRecordResult:
        os.system("echo " + name + " " + str(round(FinishTime, 2)) +
                  #   " " + str(round(Dispersion, 8)) +
                  " >> output/results.out")

    os.system("rm "+dampingPath)
    os.system("rm "+stiffnessPath)
    os.system("rm "+meshPath)

    if(return_item == ReturnType.SUCCESS):
        return Finished
    elif (return_item == ReturnType.TIME):
        return FinishTime
    elif (return_item == ReturnType.DISPERSION):
        return Dispersion


# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('parameters', metavar='N', type=float, nargs='+',
#                     help='an integer for the accumulator')

# args = parser.parse_args()
# print(args.parameters)

# ExploreLabyrinth(args.parameters[0],  args.parameters[1], reaction=5, output=OutputType.RECORD,
#                  return_item=ReturnType.TIME, verbose=False)
