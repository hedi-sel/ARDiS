from ardis import *

import ardis.d_geometry as dg

from scipy.sparse import *
import scipy.sparse.linalg as spLnal
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as colors
from matplotlib import ticker, cm
import numpy as np

# import multiprocessing as mp
import time
import math
import argparse
import os
from enum import Enum

matrixFolder = "matrixLabyrinth"  # "/mnt/ramdisk/matrixData"
printFolder = "output"
csvFolder = "outputCsv"


class OutputType(Enum):
    NONE = 0
    PLOT = 1
    RECORD = 2
    RECORD_PLOT = 3


class ReturnType(Enum):
    SUCCESS = 0
    TIME_STEP = 1
    COMPUTATION_TIME = 2
    LOADING_TIME = 3
    LOADING_COMPUTATION_TIME = 4


def PrintLabyrinth(name, verbose=True, plotEvery=1, dt=0, meshPath=""):
    if verbose:
        print("Starting result plot ...")

    if(meshPath == ""):
        meshPath = matrixFolder + "/mesh_" + name + ".dat"
    Mesh = LoadMeshFromFile(meshPath)

    Surface = (np.max(Mesh.x) - np.min(Mesh.x)) * \
        (np.max(Mesh.y) - np.min(Mesh.y))

    if verbose:
        print("Mesh loaded")

    f = open(csvFolder+"/" + name+".csv", "r")
    lines = f.readlines()

    os.system("rm -rf " + printFolder + "/" + name)
    os.system("mkdir " + printFolder + "/" + name + " 2> ./null")

    k = 0
    stride = 1 + 2  # 1 + number of species
    # stateName, nSpecies = lines.pop(0).split("\t")
    for i in range(0, len(lines) // stride):
        if (i % plotEvery == 0):
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            if dt == 0:
                title = str(i)
            else:
                title = str(round(i * dt, 1)) + "s"

            N = int(lines[i*stride])
            plt.scatter(Mesh.x, Mesh.y, marker='s', s=5 * math.sqrt(Surface *
                                                                    1.0 / N), c=[[0, 0.1, 0.3]], alpha=1, vmin=0, vmax=1)

            line = lines[i*stride+1].split("\t")
            line.pop(0)
            U = np.array(line, dtype=np.float32)
            ax.set_title(title)
            plt.scatter(Mesh.x, Mesh.y, s=U*2 * math.sqrt(Surface *
                                                          1.0 / N), c=[[0.8, 0.1, 0.2]], alpha=1, vmin=0, vmax=1)

            line = lines[i*stride+2].split("\t")
            line.pop(0)
            U = np.array(line, dtype=np.float32)
            # for j in range(0, len(U)):
            #     if (U[j] >2.5):
            #         U[j] = 2.5
            ax.set_title(title)
            plt.scatter(Mesh.x, Mesh.y, s=U * 2 * math.sqrt(Surface *
                                                            1.0 / N), c=[[0.2, 0.4, 0.6]], alpha=1, vmin=0, vmax=1)

            plt.savefig(printFolder+"/"+name+"/"+str(i)+".png")
            plt.close()

        if verbose:
            if i >= k * len(lines) / 10 / stride:
                print(str(k * 10) + "% completed")
                k += 1

    os.system("convert -delay 10 -loop 0 $(ls -1 "+printFolder +
              "/" + name + "/*png | sort -V) " + printFolder + "/" + name + ".gif" + " && " +
              "rm -rf " + printFolder + "/" + name)

    if(verbose):
        print("Results plot have been saved here: " +
              printFolder + "/" + name + ".gif")


def CompareExterioInterior(name, dt=1e-2, verbose=True):
    if verbose:
        print("Starting exterior-interior compairing ...")

    meshPath = matrixFolder + "/mesh_" + name + ".dat"
    Mesh = LoadMeshFromFile(meshPath)

    f = open(csvFolder+"/" + name+".csv", "r")
    lines = f.readlines()

    def norm(x, y):
        return math.sqrt(x * x + y * y)

    nTests = 10
    angles = np.linspace(0, math.pi/2, nTests+1)
    TestZonesExt = []
    TestZonesInt = []

    for i in range(0, nTests):
        TestZonesExt = TestZonesExt + [ConditionalZone(TriangleZone(0, 0, 3*math.cos(angles[i]), 3*math.sin(
            angles[i]), 3 * math.cos(angles[i + 1]), 3 * math.sin(angles[i + 1])), lambda x, y: norm(x, y) > 2)]
        TestZonesInt = TestZonesInt + [ConditionalZone(TriangleZone(0, 0, 3*math.cos(angles[i]), 3*math.sin(
            angles[i]), 3 * math.cos(angles[i + 1]), 3 * math.sin(angles[i + 1])), lambda x, y: norm(x, y) < 2)]
    SpeedOut = []
    TimeOfSpeedOut = []
    LastSpeedOut = 0
    currentOutTest = 0

    SpeedIns = []
    TimeOfSpeedIns = []
    LastSpeedIns = 0
    currentInsTest = 0

    threashold = 0.9

    i = -1
    for line in lines:
        i += 1
        U = np.array(line.split("\t"), dtype=np.float32)
        while (currentOutTest < nTests):
            extr, j = GetMaxZone(U, Mesh, TestZonesExt[currentOutTest]), 0
            if extr > threashold:
                if(LastSpeedOut != 0):
                    print(currentOutTest, ":", j, "/", jMin)
                    SpeedOut.append(
                        norm(Mesh.x[jMin] - Mesh.x[j], Mesh.y[jMin] - Mesh.y[j]) /
                        (i * dt - LastSpeedOut))
                    TimeOfSpeedOut.append(i*dt)
                LastSpeedOut = i*dt
                jMin = j
                currentOutTest += 1
            else:
                break
        while(currentInsTest < nTests):
            extr, j = GetMaxZone(U, Mesh, TestZonesInt[currentInsTest]), 0
            if (extr > threashold):
                if (LastSpeedIns != 0):
                    print(currentInsTest, ":", j, "/", jMax)
                    SpeedIns.append(
                        norm(Mesh.x[jMax] - Mesh.x[j], Mesh.y[jMax] - Mesh.y[j]) /
                        (i * dt - LastSpeedIns))
                    TimeOfSpeedIns.append(i*dt)
                LastSpeedIns = i*dt
                jMax = j
                currentInsTest += 1
            else:
                break
    plt.plot(TimeOfSpeedIns, SpeedIns)
    plt.plot(TimeOfSpeedOut, SpeedOut)
    # plt.show()
    print("Exteriot : ", TimeOfSpeedOut)
    print("Interior : ", TimeOfSpeedIns)


def PrepareArea(out, ins, thickness=1, name="noName"):
    name = str(round(out, 2))+"_"+str(round(ins, 2)) + \
        "_" + str(round(thickness, 2))
    os.system("./MakeMatrix.wls "+name+" "+str(out) +
              " " + str(ins) + " " + str(thickness))
    return name


def ExploreLabyrinth(name, diffusion=1, reaction=5, output=OutputType.NONE, return_item=ReturnType.SUCCESS, storeEvery=1, verbose=True, dt=1e-2, epsilon=1e-3, max_time=3, plot_dt=1e-1, drain=1e-13, guess=float(0),
                     threashold=0.9, startZone=dg.RectangleZone(0, 0, 500, 0.2), fastCalculation=False):
    print("Starting exploration on experiment :", name)
    if True:  # Prepare output type and return item
        DoPlot = False
        DoRecordResult = False
        if (output == OutputType.PLOT):
            DoPlot = True
        elif (output == OutputType.RECORD):
            DoRecordResult = True
        elif (output == OutputType.RECORD_PLOT):
            DoPlot = True
            DoRecordResult = True

    start = time.time()

    if True:  # Load Mesh and matrices from file
        dampingPath = matrixFolder+"/damping_"+name+".mtx"
        stiffnessPath = matrixFolder+"/stiffness_"+name+".mtx"
        meshPath = matrixFolder + "/mesh_" + name + ".dat"

        Mesh = LoadMeshFromFile(meshPath)

    # Remove csv file from previous calculations
    if not fastCalculation:
        os.system("rm -f "+csvFolder+"/"+name+".csv")

    d_S = ToD_SparseMatrix(LoadMatrixFromFile(
        stiffnessPath, Readtype.Symetric), MatrixType.CSR)
    if(verbose):
        print("Stiffness matrix loaded ...")
    d_D = ToD_SparseMatrix(LoadMatrixFromFile(
        dampingPath, Readtype.Symetric), MatrixType.CSR)
    if (verbose):
        print("Dampness matrix loaded ...")

    loading_time = time.time()-start
    start = time.time()

    system = System(d_D.Cols)
    system.Drain = drain
    system.Epsilon = epsilon

    n = len(Mesh.x)

    U = D_Vector(n)
    U.FillValue(0)
    Mesh = dg.D_Mesh(Mesh.x, Mesh.y)
    dg.FillZone(U, Mesh, startZone, 1)

    print(U)

    system.AddSpecies("N")
    system.SetSpecies("N", U)
    # system.AddSpecies("NP")
    # system.SetSpecies("NP", np.array([0] * len(U)))
    system.AddSpecies("P")
    system.SetSpecies("P", U)

    system.LoadStiffnessMatrix(d_S)
    system.LoadDampnessMatrix(d_D)

    system.AddMMReaction(" N -> 2 N", reaction, 1)
    # system.AddReaction(" N+P -> NP", reaction)
    # system.AddMMReaction(" NP -> 2P", reaction, 1)
    system.AddReaction("N+P-> 2P", reaction)
    # system.AddReaction(" N -> 2 N", reaction)
    # system.AddReaction(" 2N -> N", reaction)

    Nit = int(max_time / dt)
    k = 0

    if (DoRecordResult):
        FinishZone = RectangleZone(0, 0, 0.5, 500)
        FinishTime = -1

    for i in range(0, Nit):
        system.IterateDiffusion(dt)
        system.Prune()
        system.IterateReaction(dt, True)

        if not fastCalculation:
            if i % storeEvery == 0:
                ToCSV(system.State, "./"+csvFolder+"/"+name+".csv")

            if DoRecordResult and FinishTime == -1 and GetMaxZone(U, Mesh, FinishZone) > 0.8:
                FinishTime = i*dt

            if verbose:
                if Nit >= 100 and i >= k * Nit / 10 and i < k * Nit / 10 + 1:
                    print(str(k * 10) + "% completed")
                    k += 1

    print("Exploration finished in", i*dt, "s")

    computation_time = time.time() - start
    if verbose:
        print("Total computation time :", computation_time)
        system.Print()

    if not fastCalculation:

        if DoRecordResult:
            os.system("echo " + name + " " + str(round(FinishTime, 1+max(0, int(-math.log10(dt))))) +
                      " >> output/results.out")
            if(verbose):
                print("Results have been recorded here: output/results.out")

        if (DoPlot):
            # pool = mp.Pool(mp.cpu_count())
            # pool.apply(PrintLabyrinth, args=(name,  verbose, int(plot_dt / dt), dt))
            PrintLabyrinth(name, verbose=verbose, plotEvery=int(
                plot_dt/dt/storeEvery), dt=dt)

    if (return_item == ReturnType.TIME_STEP):
        return i * dt
    if (return_item == ReturnType.COMPUTATION_TIME):
        return computation_time
    if (return_item == ReturnType.LOADING_TIME):
        return loading_time
    if (return_item == ReturnType.LOADING_COMPUTATION_TIME):
        return loading_time, computation_time
