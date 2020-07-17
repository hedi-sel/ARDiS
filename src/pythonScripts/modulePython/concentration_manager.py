import numpy as np
import math


def Sign(x0, y0, x1, y1, x2, y2):
    return (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)


class ConditionalZone:
    zone, condition = 0, 0

    def __init__(self, zone, condition):
        self.zone = zone
        self.condition = condition

    def IsInside(self, x, y):
        if(self.zone.IsInside(x, y)):
            print(math.sqrt(x * x + y * y), self.condition(x, y))
        self.zone.IsInside(x, y) and self.condition(x, y)


class TriangleZone:
    x0, y0, x1, y1, x2, y2 = 0.0, 0.0, 1.0, 0.0, 1.0, 1.0

    def __init__(self, x0, y0, x1, y1, x2, y2):
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2
        self.y0 = y0
        self.y1 = y1
        self.y2 = y2

    def IsInside(self, x, y):
        b1 = Sign(x, y, self.x0, self.y0, self.x1, self.y1) < 0.0
        b2 = Sign(x, y, self.x1, self.y1, self.x2, self.y2) < 0.0
        b3 = Sign(x, y, self.x2, self.y2, self.x0, self.y0) < 0.0
        return ((b1 == b2) and (b2 == b3))


class RectangleZone:
    x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0

    def __init__(self, x0, y0, x1, y1):
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
            iM = i
    return Min


def GetMaxZone(U, Mesh, Zone):
    Max = float('-inf')
    for i in range(0, len(U)):
        if Zone.IsInside(Mesh.x[i], Mesh.y[i]) and U[i] > Max:
            Max = U[i]
            iM = i
    return Max


def GetMeanZone(U, Mesh, Zone):
    Sum = 0
    N = 0
    for i in range(0, len(U)):
        if Zone.IsInside(Mesh.x[i], Mesh.y[i]):
            Sum += U[i]
            N += 1
    return Sum/N
