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
csvFolder = "outputCsv"


class OutputType(Enum):
    NONE = 0
    PLOT = 1
    RECORD = 2


class ReturnType(Enum):
    SUCCESS = 0
    TIME = 1
    DISPERSION = 2


def PrintLabyrinth(name, verbose=True, plotEvery=1):
    if verbose:
        print("Starting result plot ...")

    meshPath = matrixFolder + "/mesh_" + name + ".dat"
    Mesh = LoadMeshFromFile(meshPath)

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
            title = str(i)
            plt.scatter(Mesh.x, Mesh.y, c=U, alpha=0.3, vmin=0, vmax=1)
            plt.savefig(printFolder+"/"+name+"/"+title+".png")
            plt.close()

    os.system("convert -delay 10 -loop 0 $(ls -1 "+printFolder +
              "/" + name+"/*png | sort -V) "+printFolder+"/"+name+".gif")
    os.system("rm -rf " + printFolder + "/" + name)

    if(verbose):
        print("Results plot have been saved here: " +
              printFolder + "/" + name + ".gif")


def RecordResults(name, dt=0.1, verbose=True):
    if verbose:
        print("Starting result recording ...")

    meshPath = matrixFolder + "/mesh_" + name + ".dat"
    Mesh = LoadMeshFromFile(meshPath)

    f = open(csvFolder+"/" + name+".csv", "r")
    lines = f.readlines()

    FinishZone = RectangleZone(0, 0, 0.2, 10)
    FinishTime = 10

    # stateName, nSpecies = lines.pop(0).split("\t")
    i = -1
    for line in lines:
        i += 1
        U = np.array(line.split("\t"), dtype=np.float32)
        if (GetMinZone(U, Mesh, FinishZone) > 0.8):
            FinishTime = i*dt
            break

    os.system("echo " + name + " " + str(round(FinishTime, 2)) +
              #   " " + str(round(Dispersion, 8)) +
              " >> output/results.out")

    if(verbose):
        print("Results have been recorded here: output/results.out")


def CppExploreLabyrinth(out, ins, thickness=1, reaction=5, output=OutputType.NONE, return_item=ReturnType.SUCCESS, verbose=True, dt=1e-2, epsilon=1e-3, max_time=3, plot_dt=1e-1, guess=float(0), use_system=True,
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

        Mesh = LoadMeshFromFile(meshPath)

        n = len(Mesh.x)

        fillVal = 0.001
        U = np.full(n, fillVal)
        StartZone = RectangleZone(0, 0, 10, 0.2)
        # StartZone = TriangleZone(-1, -1, 5, -0.8, 5, -1)
        FillZone(U, Mesh, StartZone, 1)
        print(U)

    os.system("rm -f "+printFolder+"/" + name+".csv")

    success = False

    success = dna.CppExplore(dampingPath, stiffnessPath, U,
                             reaction, max_time, dt, plot_dt, name, Mesh.x, Mesh.y)
    if not success:
        return

    start = time.time()
    k = 0
    Nit = int(max_time / dt)

    solve2Time = time.time() - start

    if(verbose):
        print("Run Time:", solve2Time)

    if DoRecordResult:
        RecordResults(name, plot_dt, verbose=verbose)

    if (DoPlot):
        PrintLabyrinth(name, verbose=verbose)
