from ardis import *
import ardis.d_geometry as dg
import ardis.geometry as geo

import numpy as np

import os


matrixFolder = "data"
outputFolder = "output"

name = "curve_1.5mm"
print("Starting exploration on experiment :", name)

drain, epsilon = 0, 1e-3
diffusion = 0.1

dt, max_time = 0.1, 10
plot_dt = 0.1


dampingPath = matrixFolder+"/"+name+"_damping.mtx"
stiffnessPath = matrixFolder+"/"+name+"_stiffness.mtx"
meshPath = matrixFolder + "/" + name + "_mesh.dat"

Mesh = dg.read_mesh(meshPath)

d_S = to_d_spmatrix(read_spmatrix(
    stiffnessPath, Readtype.Symetric), matrix_type.CSR)
print("Stiffness matrix loaded ...")
d_D = to_d_spmatrix(read_spmatrix(
    dampingPath, Readtype.Symetric), matrix_type.CSR)
print("Dampness matrix loaded ...")

st = state(d_D.shape[0])
n = len(Mesh.x)


d_Mesh = dg.d_mesh(Mesh.x, Mesh.y)

upleftcorner = dg.rect_zone(dg.point2d(0, 9), dg.point2d(1, 10))
uprightcorner = dg.rect_zone(dg.point2d(9, 9), dg.point2d(10, 10))


simu = simulation(d_D.shape[0])

startZone = dg.rect_zone(dg.point2d(0, 0), dg.point2d(0.1, 3))


simu.add_species("N")
simu.add_species("P")

dg.fill_zone(simu.get_species("N"), d_Mesh, startZone, 1)
dg.fill_zone(simu.get_species("P"), d_Mesh, startZone, 1)


simu.add_reaction(" N -> 2 N", 1)
simu.add_reaction(" N + P -> 2P", 1)

simu.drain = drain
simu.epsilon = epsilon

simu.load_stiffness_matrix(d_S)
simu.load_dampness_matrix(d_D)

Nit = int(max_time / dt)
verboseCount = 0
plotcount = 0

os.system("rm -rf " + outputFolder + "/" + name)
os.system("mkdir " + outputFolder + "/" + name)


for i in range(0, Nit):

    if (i * dt >= plot_dt * plotcount):
        fig = plot_state(simu.state, Mesh, title=str(
            plotcount))
        fig.savefig(
            outputFolder + "/"+name+"/" + str(i) + ".png")
        plt.close(fig)
        plotcount += 1

    simu.iterate_reaction(dt)
    simu.iterate_diffusion(diffusion * dt)

    if Nit >= 100 and i >= verboseCount * Nit / 10 and i < verboseCount * Nit / 10 + 1:
        print(str(verboseCount * 10) + "% completed")
        verboseCount += 1

simu.print_profiler()

os.system("convert -delay 10 -loop 0 $(ls -1 "+outputFolder +
          "/" + name + "/*png | sort -V) " + outputFolder + "/" + name + ".gif")

print("Results plot have been saved here: " +
      outputFolder + "/" + name + ".gif")
