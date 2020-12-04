from .ardisLib import state

import numpy as np
from scipy.sparse import *
import matplotlib.tri as tri
from enum import Enum


def line_to_values(line):
    if (line[0] == '%'):
        return None
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


def read_spmatrix(path, readtype=Readtype.Normal):
    f = open(path, "r")
    lines = f.readlines()
    i = -1
    line = None
    while (line == None):
        line = line_to_values(lines.pop(0))
    i, j, k = line
    mat = lil_matrix((i, j))
    for line in lines:
        values = None
        while (values == None):
            values = line_to_values(line)
        if len(values) == 3:
            mat[values[0] - 1, values[1] - 1] = values[2]
            if (readtype == Readtype.Symetric and values[1] != values[0]):
                mat[values[1] - 1, values[0] - 1] = values[2]
        else:
            print("Could not read the following line: ", line)
    return mat


def read_state(path):

    f = open(path)
    lines = f.readlines()

    vect_size, n_species = line_to_values(lines.pop(0))

    imp_state = state(vect_size)
    species_list = {}

    for i in range(0, n_species):
        species_idx, species_name = line_to_values(lines.pop(0))
        species_list[species_idx] = species_name

    for i in range(o, n_species):
        state.add_species(species_list[i])

    for i in range(o, n_species):
        state.add_species(species_list[i])

    for i in range(o, n_species):
        vect = lines.pop(0)
        species = vect.pop(0)
        vect = np.array(vect)
        state.get_species().fill(vect)
    return state
