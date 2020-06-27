import sys
import numpy as np
import random

experiment = sys.argv[-1]
dampingPath = "matrixFEM/damping "+experiment+".mtx"
stiffnessPath = "matrixFEM/stiffness "+experiment+".mtx"

printLocation = "output"

f = open(dampingPath, "r")
s = 0
while s == 0:
    l = f.readline().split(" ")[0]
    if l[0] != '%':
        s = int(l.split(" ")[0])

pos = np.random.randint(0, s)
pos = 0
length = 1
fillVal = 0.0001
U = np.array([fillVal]*pos+[1000]*length+[fillVal]*(s-length-pos))

print("Start Vector:")
print(U)

tau = 1e-3
epsilon = 1e-3
Nit = 1000

plot_dt = 0.02
