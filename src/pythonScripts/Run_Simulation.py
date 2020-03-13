from CppLabyrinthExplorer import *
import math
import numpy as np
import os

Out = np.logspace(math.log10(1.5), math.log10(15), 100)
Ins = np.logspace(math.log10(0.75), math.log10(7.5), 100)
Thi = np.linspace(0.1, 1, 11)
Rea = 5  # np.logspace(math.log10(1), math.log10(10), 10)

# CppExploreLabyrinth(2.5/4, 0.5/4, thickness=0.5, reaction=5, output=OutputType.PLOT, plot_dt=0.1, dt=0.01, max_time=10,
#                     return_item=ReturnType.TIME, verbose=True, use_system=True)

# CppExploreLabyrinth(9./4, 7./4, thickness=0.5, reaction=5, output=OutputType.PLOT, plot_dt=0.1, dt=0.01, max_time=10,
#                     return_item=ReturnType.TIME, verbose=True, use_system=True)

# CppExploreLabyrinth(Out[0], Ins[0], thickness=Thi[3], reaction=Rea, output=OutputType.PLOT,
#                     return_item=ReturnType.TIME, verbose=False)


for out in Out:
    for ins in Ins:
        guess = 0.0
        for thi in Thi:
            CppExploreLabyrinth(out, ins, thickness=thi, reaction=Rea, output=OutputType.NONE,
                                return_item=ReturnType.TIME, verbose=False)

# PrintLabyrinth("1.5_0.8_0.1_5")
# PrintLabyrinth("1.5_0.8_0.37_5")
# PrintLabyrinth("1.5_0.8_1.0_5")
