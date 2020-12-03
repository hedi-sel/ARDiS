from ..ardisLib.d_geometry import *
import matplotlib.tri as tri
import numpy as np


def to_d_mesh(mesh):
    return to_d_mesh(mesh.x, mesh.y)
