from ardis import *
import ardis.d_geometry as dg
import ardis.geometry as geo

import numpy as np

import os

matrixFolder = "data"
outputFolder = "output"

name = "coarse_square"
print("Starting exploration on experiment :", name)

drain, epsilon = 0, 1e-3
diffusion = 1
production = 100

dt, max_time = 0.1, 120
plot_dt = 2


dampingPath = matrixFolder+"/"+name+"_damping.mtx"
stiffnessPath = matrixFolder+"/"+name+"_stiffness.mtx"
meshPath = matrixFolder + "/" + name + "_mesh.dat"

Mesh = dg.read_mesh(meshPath)

d_S = to_d_spmatrix(read_spmatrix(
    stiffnessPath, read_type.Symetric), matrix_type.CSR)
print("Stiffness matrix loaded ...")
d_D = to_d_spmatrix(read_spmatrix(
    dampingPath, read_type.Symetric), matrix_type.CSR)
print("Dampness matrix loaded ...")

n = len(Mesh.x)


d_Mesh = dg.d_mesh(Mesh.x, Mesh.y)

upleftcorner = dg.rect_zone(dg.point2d(0, 9), dg.point2d(1, 10))
uprightcorner = dg.rect_zone(dg.point2d(9, 9), dg.point2d(10, 10))


simu = simulation(n)
import_crn(simu, "chemicalReactionNetworks/bottomMETI.json")

simu.drain = drain
simu.epsilon = epsilon

simu.load_stiffness_matrix(d_S)
simu.load_dampness_matrix(d_D)

Nit = int(max_time / dt)
verboseCount = 0
plotcount = 0

os.system("rm -rf " + outputFolder + "/" + name)
os.system("mkdir " + outputFolder + "/" + name)

dg.fill_zone(simu.get_species("0"), d_Mesh, upleftcorner, production)
dg.fill_zone(simu.get_species("1"), d_Mesh, uprightcorner, production)

for i in range(0, Nit):

    dg.fill_zone(simu.get_species("0"), d_Mesh, upleftcorner, production)
    dg.fill_zone(simu.get_species("1"), d_Mesh, uprightcorner, production)

    if (i * dt >= plot_dt * plotcount):
        fig = plot_state(simu.state, Mesh, title=str(plotcount), listSpecies=["0", "1",
                                                                              "2"], colors={"background": (0, 0, 0)})
        fig.savefig(
            outputFolder + "/"+name+"/" + str(i) + ".png")
        plt.close(fig)
        plotcount += 1

    simu.iterate_reaction(dt)
    simu.prune(0)
    # simu.prune_under(1)
    simu.iterate_diffusion(diffusion * dt)

    # simu.get_species("2").print(10000)

    if Nit >= 100 and i >= verboseCount * Nit / 10 and i < verboseCount * Nit / 10 + 1:
        print(str(verboseCount * 10) + "% completed")
        verboseCount += 1

simu.print_profiler()

os.system("convert -delay 10 -loop 0 $(ls -1 "+outputFolder +
          "/" + name + "/*png | sort -V) " + outputFolder + "/" + name + ".gif")

print("Results plot have been saved here: " +
      outputFolder + "/" + name + ".gif")
