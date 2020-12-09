from ardis import *
import ardis.d_geometry as dg
import ardis.geometry as geo

import numpy as np

import os

matrixFolder = "data"
outputFolder = "output"

reaction, drain, epsilon = 1,  1e-8, 1e-3

diffusion = 10

dt, max_time = 0.1, 1000

plot_dt = 10

name = "square"

print("Starting exploration on experiment :", name)

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

upleftcorner = dg.rect_zone(dg.point2d(90, 0), dg.point2d(100, 10))
uprightcorner = dg.rect_zone(dg.point2d(90, 90), dg.point2d(100, 100))


simu = simulation(d_D.shape[0])
import_crn(simu, "chemicalReactionNetworks/t_approx.json")

simu.drain = drain
simu.epsilon = epsilon

simu.load_stiffness_matrix(d_S)
simu.load_dampness_matrix(d_D)

Nit = int(max_time / dt)
verboseCount = 0
plotcount = 0

os.system("mkdir " + outputFolder + "/" + name)

for i in range(0, Nit):

    dg.fill_zone(simu.get_species("0"), d_Mesh, upleftcorner, 1)
    dg.fill_zone(simu.get_species("1"), d_Mesh, uprightcorner, 1)

    simu.iterate_diffusion(diffusion * dt)
    simu.prune()
    simu.iterate_reaction(dt, True)

    if (i * dt > plot_dt * plotcount):
        plotcount += 1
        fig = plot_state(simu.state, Mesh, listSpecies=[
                         "0", "1", "2"], colors={"2": [[1, 0, 0]], "background": [0, 0, 0]})
        fig.savefig(
            outputFolder + "/"+name+"/" + str(i) + ".png")
        plt.close(fig)

    if Nit >= 100 and i >= verboseCount * Nit / 10 and i < verboseCount * Nit / 10 + 1:
        print(str(verboseCount * 10) + "% completed")
        verboseCount += 1

os.system("convert -delay 10 -loop 0 $(ls -1 "+outputFolder +
          "/" + name + "/*png | sort -V) " + outputFolder + "/" + name + ".gif" + " && " +
          "rm -rf " + outputFolder + "/" + name)

print("Results plot have been saved here: " +
      outputFolder + "/" + name + ".gif")
