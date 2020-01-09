import numpy as np
from scipy.sparse import *
from enum import Enum

import modulePython.dna as dna


def ReadLine(line):
    if (line[0] == '%'):
        return -1, -1, 0
    values = []
    for str in line.split(" "):
        if "." in str or "e" in str:
            values.append(float(str))
        else:
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
        mat[values[0] - 1, values[1] - 1] = values[2]
        if (readtype == Readtype.Symetric):
            mat[values[1] - 1, values[0] - 1] = values[2]
    return mat


# def LoadMatrixFromFileToGPU(path):
#     f = open(path, "r")
#     lines = f.readlines()
#     i = -1
#     while (i == -1):
#         i, j, k = ReadLine(lines.pop(0))

#     M = dna.MatrixSparse(i, j, k, dna.COO, False)
#     k = 0
#     for line in lines:
#         values = ReadLine(line)
#         M.AddElement(k, values[0] - 1, values[1] - 1, values[2])
#         k += 1
#     return M
