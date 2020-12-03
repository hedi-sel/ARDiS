import numpy as np
from scipy.sparse import *
import matplotlib.tri as tri
from enum import Enum
from ..convert import readline


def read_mesh(path):
    f = open(path, "r")
    lines = f.readlines()
    x,  y = [], []
    for line in lines:
        values = readline(line)
        if len(values) == 2:
            x.append(values[0])
            y.append(values[1])
        else:
            print("Could not read the following line: ", line)

    return tri.Triangulation(np.array(x), np.array(y))


def write_mesh(path):
    print("write_mesh function has not been implemented")
