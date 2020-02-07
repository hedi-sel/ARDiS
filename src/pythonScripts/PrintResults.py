import numpy as np
import matplotlib.pyplot as plt

resultsLocation = "output/results.out"

f = open(resultsLocation, "r")
lines = f.readlines()

######################
# A modifier
#######
Z = np.zeros((10, 10))
nvals = np.zeros((10, 10))
xparam = 0
yparam = 1
zresult = 0


x, y = 0, 0
xprev, yprev = 0, 0
i, j = -1, -1
maxval = 0
for line in lines:
    elmts = line.split(" ")
    params = elmts[0].split("_")
    xprev = x
    x = float(params[xparam])
    yprev = y
    y = float(params[yparam])
    if (xprev < x):
        i += 1
    elif (xprev > x):
        i = 0
    if (yprev < y):
        j += 1
    elif (yprev > y):
        j = 0
    Z[i, j] += float(elmts[1 + zresult])
    nvals[i, j] += 1
    if (maxval < Z[i, j]):
        maxval = Z[i, j]
Z /= maxval
nvals += 0.01
Z = Z / nvals


Z = Z / (Z.max(axis=0).max(axis=0) + np.spacing(0))
plt.imshow(Z)

plt.savefig("output/print.png")
plt.close()
