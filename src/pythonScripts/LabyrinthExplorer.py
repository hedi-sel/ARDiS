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

matrixFolder = "/mnt/ramdisk/matrixData" #"matrixLabyrinth"
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
    LOADING_COMPUTATION_TIME=4


def PrintLabyrinth(name, verbose=True, plotEvery=1, dt=0, meshPath=""):
    if verbose:
        print("Starting result plot ...")

    if(meshPath == ""):
        meshPath = matrixFolder + "/mesh_" + name + ".dat"
    Mesh = LoadMeshFromFile(meshPath)
    Surface = (np.max( Mesh.x)-np.min( Mesh.x)) * (np.max( Mesh.y) - np.min( Mesh.y))

    f = open(csvFolder+"/" + name+".csv", "r")
    lines = f.readlines()

    os.system("rm -rf "+printFolder+"/"+name)
    os.system("mkdir " + printFolder + "/" + name + " 2> ./null")

    # stateName, nSpecies = lines.pop(0).split("\t")
    i = -1
    for line in lines:
        i += 1
        if(i % plotEvery == 0):
            U = np.array(line.split("\t"), dtype=np.float32)
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            if dt == 0:
                title = str(i)
            else:
                title = str(round(i * dt, 1)) + "s"
            ax.set_title(title)
            plt.scatter(Mesh.x, Mesh.y, s=10*Surface*1.0/len(U), c=U, alpha=0.5, vmin=0, vmax=1)
            plt.savefig(printFolder+"/"+name+"/"+str(i)+".png")
            plt.close()

    os.system("convert -delay 10 -loop 0 $(ls -1 "+printFolder +
              "/" + name+"/*png | sort -V) "+printFolder+"/"+name+".gif")
    os.system("rm -rf " + printFolder + "/" + name)

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


def ExploreLabyrinth(name, diffusion=1, reaction=5, output=OutputType.NONE, return_item=ReturnType.SUCCESS, verbose=True, dt=1e-2, epsilon=1e-3, max_time=3, plot_dt=1e-1, guess=float(0),
                     threashold=0.9, startZone=RectangleZone(0, 0, 500, 0.2), fastCalculation = False):
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

        n = len(Mesh.x)

        fillVal = 0
        U = np.full(n, fillVal)
        FillZone(U, Mesh, startZone, 1)
        if verbose:
            print("Starting vector :", U)

    # Remove csv file from previous calculations
    if not fastCalculation:
        os.system("rm -f "+csvFolder+"/"+name+".csv")

    d_S = rd.ToD_SparseMatrix(LoadMatrixFromFile(
        stiffnessPath, Readtype.Symetric), dna.MatrixType.CSR)
    if(verbose):
        print("Stiffness matrix loaded ...")
    d_D = rd.ToD_SparseMatrix(LoadMatrixFromFile(
        dampingPath, Readtype.Symetric), dna.MatrixType.CSR)
    if (verbose):
        print("Dampness matrix loaded ...")
    
    loading_time = time.time()-start
    start = time.time()

    system = dna.System(d_D.cols)
    system.SetEpsilon(epsilon)
    system.AddSpecies("N")
    system.SetSpecies("N", U)
    system.AddSpecies("P")
    system.SetSpecies("P", U)

    system.LoadStiffnessMatrix(d_S)
    system.LoadDampnessMatrix(d_D)

    system.AddReaction(" N -> 2 N", reaction)
    system.AddReaction(" N+P -> 2P", reaction)
    # system.AddReaction(" N -> 2 N", reaction)
    # system.AddReaction(" 2N -> N", reaction)

    Nit = int(max_time / dt)
    k = 0

    if (DoRecordResult):
        FinishZone = dna.RectangleZone(0, 0, 0.5, 500)
        FinishTime = -1
        d_MeshX = dna.D_Array(len(Mesh.x))
        d_MeshX.Fill(Mesh.x)
        d_MeshY = dna.D_Array(len(Mesh.y))
        d_MeshY.Fill(Mesh.y)

    for i in range(0, Nit):
        system.IterateDiffusion(dt)
        system.Prune()
        system.IterateReaction(dt, True)

        if not fastCalculation:
            dna.ToCSV(system.State, "N", "./"+csvFolder+"/"+name+".csv")

            if DoRecordResult and FinishTime == -1 and GetMaxZone(U,Mesh,FinishZone) > 0.8:
            # if DoRecordResult and dna.zones.GetMinZone(d_U, d_MeshX, d_MeshY, FinishZone) > 0.8:
                FinishTime = i*dt

            if verbose:
                if Nit >= 100 and i >= k * Nit / 10 and i < k * Nit / 10 + 1:
                    print(str(k * 10) + "% completed")
                    k += 1

        if (i > 0 and (system.GetSpecies("N") - d_U).Norm() < 1.e-3):
            print(i+1, "iterations done")
            break
        d_U = dna.D_Array(system.GetSpecies("N"))
        U = d_U.ToNumpyArray()

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
            PrintLabyrinth(name, verbose=verbose, plotEvery=int(plot_dt/dt), dt=dt)

    if (return_item == ReturnType.TIME_STEP):
        return i * dt
    if (return_item == ReturnType.COMPUTATION_TIME):
        return computation_time
    if (return_item == ReturnType.LOADING_TIME):
        return loading_time
    if (return_item == ReturnType.LOADING_COMPUTATION_TIME):
        return loading_time, computation_time