import numpy as np


class RectangleZone:
    x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0

    def __init__(self, x0, x1, y0, y1):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

    def IsInside(self, x, y):
        return self.x0 <= x and self.x1 >= x and self.y0 <= y and self.y1 >= y


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
