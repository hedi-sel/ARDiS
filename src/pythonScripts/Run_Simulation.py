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
#                     return_item=ReturnType.TIME, verbose=True,  UseCpp=False)
# CompareExterioInterior(name)

# name = PrepareArea(9./4, 7./4, thickness=0.5)
# ExploreLabyrinth(name, reaction=5, output=OutputType.PLOT, plot_dt=0.1, dt=0.01, max_time=10,
#                     return_item=ReturnType.TIME, verbose=True, UseCpp=False)
# CompareExterioInterior(name)

# ExploreLabyrinth(PrepareArea(Out[0], Ins[0], thickness=Thi[3]), reaction=Rea, output=OutputType.PLOT,
#                     return_item=ReturnType.TIME, verbose=False)

# os.system("./ImageToMatrix.wls maze_e-2_e-3")
# ExploreLabyrinth("maze_e-2_e-3", output=OutputType.PLOT,
#                  max_time=1000, diffusion=1, reaction=5, dt=1e-2, epsilon=1e-3, plot_dt=1)
for out in Out:
    if startOut > 0:
        startOut -= 1
        continue
    for ins in Ins:
        if startIns > 0:
            startIns -= 1
            continue
        for thi in Thi:
            if startThi > 0:
                startThi -= 1
                continue
            ExploreLabyrinth(PrepareArea(out, ins, thickness=thi),
                             reaction=Rea, output=OutputType.RECORD_PLOT, verbose=True)
print("fini")


# PrintLabyrinth("1.5_0.8_0.1_5")
# PrintLabyrinth("1.5_0.8_0.37_5")
# PrintLabyrinth("1.5_0.8_1.0_5")
