import sys
import numpy as np
import os
#from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

outputName = sys.argv[-1]

if (outputName == ""):
    dataLocation = input("Data Location: (default: ./output/default)")
    if (dataLocation == ""):
        dataLocation = "./output/default"
    printLocation = input("Plot Location: (default: ./plot/default)")
    if (printLocation == ""):
        printLocation = "./plot/default"
else:
    dataLocation = "./output/"+outputName
    printLocation = "./plot/"+outputName


def readLine(line):
    values = []
    nanCount = 0
    infCount = 0
    for str in line.split("\t"):
        if "." in str or "e" in str:
            values.append(float(str))
        elif "nan" in str:
            values.append(-1)
            nanCount += 1
        elif "inf" in str:
            values.append(-2)
            infCount += 1
        else:
            values.append(int(str))
    return values, nanCount, infCount


def plotAndPrintData(fileName):
    nanCount = 0
    infCount = 0
    f = open(dataLocation+"/"+fileName, "r")
    lines = f.readlines()

    """
    Expected format for the file is:
    _________
    shape (ex: 2 1024)
    0	0	val
    ..
    i	j	val
    ..
    _________
    """
    shape = readLine(lines.pop(0))[0]
    shape[2] = 3 #because rgb
    Z = np.zeros(tuple(shape))
    for line in lines:
        values, nan, inf = readLine(line)
        nanCount += nan
        infCount += inf
        z = values.pop()
        Z[tuple(values)] = z

    if (len(shape) == 2):
        X = np.linspace(1, shape[1], shape[1])
        plt.plot(X, Z[0, :], label='Prey')
        plt.plot(X, Z[1, :], label='Predator')
        #plt.xticks(X/ 1000.)
        plt.legend(loc = 'upper right')
        plt.ylabel('Species concentration (a.u.)')
        plt.xlabel('x (mm)')
        plt.grid(False)
        plt.ylim((0, 4))
    elif (len(shape) == 3):
        #Z = Z / (Z.max(axis = 0).max(axis=0) + np.spacing(0))
        plt.imshow(Z)
    else:
        print("Shape not supported")
        return

    plt.savefig(printLocation+"/"+fileName.replace(".dat", ".png"))
    plt.close()

    if nanCount > 0:
        print("Warning, there is ", nanCount, " nan values")

    if infCount > 0:
        print("Warning, there is ", infCount, " infinite values")


if os.path.exists(printLocation):
    keepgoing = input(
        "The files already exist, you wanna overwrite?\ny to overwrite, n to abort, any other key to save in a another folder\n")
    if (keepgoing == "y"):
        for file in os.listdir(printLocation):
            if ".png" in file:
                os.remove(printLocation + "/" + file)
    elif (keepgoing == "n"):
        sys.exit("Left the program without plotting anything")
    else:
        i = 2
        while os.path.exists(printLocation + "_" + str(i)):
            i += 1
        os.makedirs(printLocation + "_" + str(i))
        printLocation = printLocation + "_" + str(i)
else:
    os.makedirs(printLocation)

for file in os.listdir(dataLocation):
    plotAndPrintData(file)

os.system("./makegif.sh "+outputName)



# ax = fig.add_subplot(111, projection='3d')
# x, y = np.array(2), np.array(128)
# X, Y = np.meshgrid(x, y)

# ax.plot_surface(X, Y, Z)

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()
