from LabyrinthExplorer import *
import math
import numpy as np
import os

Out = np.logspace(math.log10(0.2), math.log10(2), 5)
Ins = np.logspace(math.log10(0.1), math.log10(1), 5)
Thi = np.linspace(0.1, 1, 5)

startOut = 0
startIns = 0
startThi = 0

Rea = 5  # np.logspace(math.log10(1), math.log10(10), 10)

# name = PrepareArea(2.5/4, 0.5/4, thickness=0.5)
# ExploreLabyrinth(name, reaction=5, output=OutputType.PLOT, plot_dt=0.1, dt=0.01, max_time=10,
#                     return_item=ReturnType.TIME_STEP, verbose=True)
# CompareExterioInterior(name)

# name = PrepareArea(9./4, 7./4, thickness=0.5)
# ExploreLabyrinth(name, reaction=5, output=OutputType.PLOT, plot_dt=0.1, dt=0.01, max_time=10,
#                     return_item=ReturnType.TIME_STEP, verbose=True)
# CompareExterioInterior(name)

# os.system("./ImageToMatrix.wls maze_e-2_e-3")
# ExploreLabyrinth("maze_e-2_e-3", output=OutputType.PLOT,
#                  max_time=20, diffusion=1, reaction=5, dt=1e-2, epsilon=1e-3, plot_dt=1, startZone=RectangleZone(0, 0, 500, 10))

# size = "100"
#  os.system("./MazeGenerator.wls maze_"+size+" "+size)
# ExploreLabyrinth("maze_"+size, output=OutputType.PLOT,
#                  max_time=10, diffusion=1, reaction=5, dt=1e-2, epsilon=1e-3, plot_dt=1, startZone=RectangleZone(0, 0, 10, 10))

Nexpr = 1
# stepSizes = ["100", "50", "30", "20", "15", "10", "5", "3", "2", "1", "0.5", "0.2"]
stepSizes = ["100", "30","2","0.2"]
# stepSizes = ["0.2"]
# stepSizes = ["0.1","0.05","0.02"]
load_times=[]
cmpt_times=[]
for step in stepSizes:
    # os.system("./ImageToMatrixControlPrecision.wls precision=" + step + " " + step)
    cmpt_time_sum= 0
    load_time_sum= 0
    for j in range(0, Nexpr):
        perf_times=ExploreLabyrinth("precision="+step, output=OutputType.PLOT, return_item=ReturnType.LOADING_COMPUTATION_TIME, verbose = True, fastCalculation = False,
                    max_time=50, diffusion=1, reaction=0.2, dt=1e-1, epsilon=1e-3, plot_dt=5, startZone=RectangleZone(0, 0, 500, 10))
        load_time_sum += perf_times[0]
        cmpt_time_sum+=perf_times[1]
    load_times.append(load_time_sum * 1.0 / Nexpr)
    cmpt_times.append(cmpt_time_sum * 1.0 / Nexpr)
    print("Load Time:", load_time_sum * 1.0 / Nexpr)
    print("Compute Time:", cmpt_time_sum * 1.0 / Nexpr)
    # input()
print ("Loading Times :",load_times)
print ("Computation Times :",cmpt_times)

# step = "0.2"
# os.system("./ImageToMatrixControlPrecision.wls precision=" + step + " " + step)
# ExploreLabyrinth("precision="+step, output=OutputType.PLOT, return_item=ReturnType.COMPUTATION_TIME, verbose = True,
#             max_time=5000, diffusion=1, reaction=0.2, dt=1e-2, epsilon=1e-2, plot_dt=5, startZone=RectangleZone(0, 0, 500, 10))

# for out in Out:
#     if startOut > 0:
#         startOut -= 1
#         continue
#     for ins in Ins:
#         if startIns > 0:
#             startIns -= 1
#             continue
#         for thi in Thi:
#             if startThi > 0:
#                 startThi -= 1
#                 continue
#             ExploreLabyrinth(PrepareArea(out, ins, thickness=thi),
#                              reaction=Rea, output=OutputType.RECORD, verbose=True)
# print("fini")


# PrintLabyrinth("1.5_0.8_0.1_5")
# PrintLabyrinth("1.5_0.8_0.37_5")
# PrintLabyrinth("1.5_0.8_1.0_5")
