import numpy as np
from scipy.sparse import *
import matplotlib.tri as tri
from enum import Enum


def ReadLine(line):
    if (line[0] == '%'):
        return -1, -1, 0
    values = []
    splitString = line.split(" ")
    if(len(splitString) == 1):
        splitString = line.split("\t")
    for str in splitString:
        if "." in str or "e" in str:
            values.append(float(str))
        elif len(str) > 0:
            values.append(int(str))
    return values


class Readtype(Enum):
    Normal = 0
    Symetric = 1


def LoadMatrixFromFile(path, readtype=Readtype.Normal):
    f = open(path, "r")
    lines = f.readlines()
    i = -1
    while (i == -1):
        i, j, k = ReadLine(lines.pop(0))
    mat = lil_matrix((i, j))
    for line in lines:
        values = ReadLine(line)
        if len(values) == 3:
            mat[values[0] - 1, values[1] - 1] = values[2]
            if (readtype == Readtype.Symetric and values[1] != values[0]):
                mat[values[1] - 1, values[0] - 1] = values[2]
        else:
            print("Could not read the following line: ", line)
    return mat


def LoadMeshFromFile(path, readtype=Readtype.Normal):
    f = open(path, "r")
    lines = f.readlines()
    x,  y = [], []
    for line in lines:
        values = ReadLine(line)
        if len(values) == 2:
            x.append(values[0])
            y.append(values[1])
        else:
            print("Could not read the following line: ", line)

    return tri.Triangulation(np.array(x), np.array(y))
