import numpy as np
import math

from ..ardisLib.geometry import *


def sign(x0, y0, x1, y1, x2, y2):
    return (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)


def fill_zone(U, Mesh, zone, value):
    for i in range(0, len(U)):
        if zone.is_inside(Mesh.x[i], Mesh.y[i]):
            U[i] = value


def fill_outside_zone(U, Mesh, zone, value):
    for i in range(0, len(U)):
        if not zone.is_inside(Mesh.x[i], Mesh.y[i]):
            U[i] = value


def fill(U, Mesh, value):
    for i in range(0, len(U)):
        U[i] = value


def min_zone(U, Mesh, zone):
    Min = float('inf')
    for i in range(0, len(U)):
        if zone.is_inside(Mesh.x[i], Mesh.y[i]) and U[i] < Min:
            Min = U[i]
    return Min


def max_zone(U, Mesh, zone):
    Max = float('-inf')
    for i in range(0, len(U)):
        if zone.is_inside(Mesh.x[i], Mesh.y[i]) and U[i] > Max:
            Max = U[i]
    return Max


def mean_zone(U, Mesh, zone):
    Sum = 0
    N = 0
    for i in range(0, len(U)):
        if zone.is_inside(Mesh.x[i], Mesh.y[i]):
            Sum += U[i]
            N += 1
    return Sum/N
