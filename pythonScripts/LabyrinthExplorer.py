from ardis import *

import ardis.d_geometry as dg
import ardis.geometry as geo

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
    Mesh = dg.read_mesh(meshPath)

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


def ExploreLabyrinth(name, diffusion=1, reaction=5, storeEvery=1, dt=1e-2, epsilon=1e-3, max_time=3, plot_dt=1e-1, drain=1e-13, guess=float(0),
                     threashold=0.9, startZone=dg.rect_zone(0, 0, 500, 0.2)):

    print("Starting exploration on experiment :", name)

    dampingPath = matrixFolder+"/damping_"+name+".mtx"
    stiffnessPath = matrixFolder+"/stiffness_"+name+".mtx"
    meshPath = matrixFolder + "/mesh_" + name + ".dat"

    Mesh = dg.read_mesh(meshPath)

    # Remove csv file from previous calculations
    os.system("rm -f "+csvFolder+"/"+name+".csv")

    d_S = to_d_spmatrix(read_spmatrix(
        stiffnessPath, Readtype.Symetric), matrix_type.CSR)
    print("Stiffness matrix loaded ...")
    d_D = to_d_spmatrix(read_spmatrix(
        dampingPath, Readtype.Symetric), matrix_type.CSR)
    print("Dampness matrix loaded ...")

    simu = simulation(d_D.shape[0])
    simu.drain = drain
    simu.epsilon = epsilon

    n = len(Mesh.x)

    U = d_vector(n)
    U.fill_value(0)
    d_Mesh = dg.d_mesh(Mesh.x, Mesh.y)
    dg.fill_zone(U, d_Mesh, startZone, 1)

    print(U)

    simu.add_species("N")
    simu.set_species("N", U)
    # simu.add_species("NP")
    # simu.set_species("NP", np.array([0] * len(U)))
    simu.add_species("P")
    simu.set_species("P", U)

    simu.load_stiffness_matrix(d_S)
    simu.load_dampness_matrix(d_D)

    simu.add_mm_reaction(" N -> 2 N", reaction, 1)
    # simu.add_reaction(" N+P -> NP", reaction)
    # simu.add_mm_reaction(" NP -> 2P", reaction, 1)
    simu.add_reaction("N+P-> 2P", reaction)
    # simu.add_reaction(" N -> 2 N", reaction)
    # simu.add_reaction(" 2N -> N", reaction)

    Nit = int(max_time / dt)
    k = 0
    plotcount = 0

    os.system("mkdir " + printFolder + "/"+name)
    # colors = {
    #     "N": [[0.8, 0.1, 0.2]],
    #     "P": [[0.2, 0.4, 0.6]]
    # }
    for i in range(0, Nit):
        simu.iterate_diffusion(dt)
        simu.prune()
        simu.iterate_reaction(dt, True)

        if (i * dt > plot_dt * plotcount):
            plotcount += 1
            fig = plot_state(simu.state, Mesh)
            fig.savefig(
                printFolder + "/"+name+"/" + str(i) + ".png")
            plt.close(fig)

        if Nit >= 100 and i >= k * Nit / 10 and i < k * Nit / 10 + 1:
            print(str(k * 10) + "% completed")
            k += 1

    os.system("convert -delay 10 -loop 0 $(ls -1 "+printFolder +
              "/" + name + "/*png | sort -V) " + printFolder + "/" + name + ".gif" + " && " +
              "rm -rf " + printFolder + "/" + name)

    print("Results plot have been saved here: " +
          printFolder + "/" + name + ".gif")
