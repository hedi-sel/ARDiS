import numpy as np
from scipy.sparse import *
from enum import Enum


def ReadLine(line):
    if (line[0] == '%'):
        return -1, -1, 0
    values = []
    for str in line.split(" "):
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
            print(line)
    return mat
