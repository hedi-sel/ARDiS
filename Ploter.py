import sys
import numpy as np
import os
import math
#from scipy.integrate import odeint
import matplotlib.pyplot as plt

resultsLocation = "output/results.out"


f = open(resultsLocation, "r")
lines = f.readlines()


nbrParams = len((lines[0].split(" "))[0].split("_"))
listOfParameters = [[]]*nbrParams
sumValForEachParam = [[]]*nbrParams
numberValuesForEachParam = [[]]*nbrParams

# for line in lines:
#     lineSplit = line.split(" ")
#     params = lineSplit[0].split("_")
#     value = float(lineSplit[1])

#     if value == -1:
#         continue

#     for i in range(0, nbrParams):
#         if params[i] not in listOfParameters[i]:
#             listOfParameters[i] = listOfParameters[i] + [params[i]]
#             sumValForEachParam[i] = sumValForEachParam[i]+[value]
#             numberValuesForEachParam[i] = numberValuesForEachParam[i] + [1]
#         else:
#             j = listOfParameters[i].index(params[i])
#             sumValForEachParam[i][j] += value
#             numberValuesForEachParam[i][j] += 1

# paramNames = ["OutsideRadius", "InsideRadius", "Thickness"]
# for i in range(0, nbrParams):
#     plt.title(paramNames[i])

#     plt.plot(np.array(listOfParameters[i]), np.array(
#         sumValForEachParam[i]) / np.array(numberValuesForEachParam[i]))

#     plt.xlabel(paramNames[i])
#     plt.ylabel('Arrival Time')

#     plt.savefig(paramNames[i] + ".png")
#     plt.close()


colorForEachInput = []
valForEachInput = []
inputList = []


for line in lines:
    lineSplit = line.split(" ")
    params = lineSplit[0].split("_")
    value = float(lineSplit[1])

    if value == -1:
        continue

    inp = float(params[0]) * (1 - math.sqrt(2)) - float(params[1]) * (1 - math.sqrt(2)) + float(params[2]) * math.sqrt(2)

    if (inp < 0):
        continue
    inputList.append(inp)
    valForEachInput.append(value)
    colorForEachInput.append( (float(params[0])/2,float(params[1]),float(params[2])) )

plt.title("Goulot d'etranglement")

plt.scatter(np.array(inputList), np.array(valForEachInput),c=np.array(colorForEachInput))
plt.xlabel("Epaisseur goulot")
plt.ylabel('Arrival Time')

plt.savefig("Goulot d'etranglement.png")
plt.close()
