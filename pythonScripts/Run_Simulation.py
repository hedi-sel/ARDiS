from LabyrinthExplorer import *

stepSizes = ["100"]
# stepSizes = ["100", "50", "30", "20", "15",
#              "10", "5", "3", "2", "1", "0.5", "0.2"]
# stepSizes = ["0.1","0.05","0.02"]
load_times = []
cmpt_times = []
for step in stepSizes:
    os.system(
        "./wolframScripts/ImageToMatrixControlPrecision.wls precision=" + step + " " + step)
    ExploreLabyrinth("precision="+step, storeEvery=1, max_time=100, diffusion=1, reaction=1,
                     dt=1e-1, drain=1e-8, epsilon=1e-3, plot_dt=1, startZone=dg.rect_zone(0, 0, 500, 10))
