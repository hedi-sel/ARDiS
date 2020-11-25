import numpy as np
import math

from ..ardisLib.geometry import *


def Sign(x0, y0, x1, y1, x2, y2):
    return (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)


def FillZone(U, Mesh, Zone, value):
    for i in range(0, len(U)):
        if Zone.IsInside(Mesh.x[i], Mesh.y[i]):
            U[i] = value


def FillOutsideZone(U, Mesh, Zone, value):
    for i in range(0, len(U)):
        if not Zone.IsInside(Mesh.x[i], Mesh.y[i]):
            U[i] = value


def Fill(U, Mesh, value):
    for i in range(0, len(U)):
        U[i] = value


def GetMinZone(U, Mesh, Zone):
    Min = float('inf')
    for i in range(0, len(U)):
        if Zone.IsInside(Mesh.x[i], Mesh.y[i]) and U[i] < Min:
            Min = U[i]
    return Min


def GetMaxZone(U, Mesh, Zone):
    Max = float('-inf')
    for i in range(0, len(U)):
        if Zone.IsInside(Mesh.x[i], Mesh.y[i]) and U[i] > Max:
            Max = U[i]
    return Max


def GetMeanZone(U, Mesh, Zone):
    Sum = 0
    N = 0
    for i in range(0, len(U)):
        if Zone.IsInside(Mesh.x[i], Mesh.y[i]):
            Sum += U[i]
            N += 1
    return Sum/N
