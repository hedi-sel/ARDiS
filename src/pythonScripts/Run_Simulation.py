from LabyrinthExplorer import ExploreLabyrinth
import math
import numpy as np

out = np.logspace(math.log10(1.5), math.log10(15), 10)
ins = np.logspace(math.log10(0.75), math.log10(7.5), 10)
reac = np.logspace(math.log10(1), math.log10(10), 10)

# for i in range(0, 10):
#     for j in range(0, 10):
#         for k in range(0, 10):
#             ExploreLabyrinth(out[i], ins[j], reaction=reac[k],
#                              output="RecordResult", verbose=False)

ExploreLabyrinth(1, 2, output="Plot", verbose=True)