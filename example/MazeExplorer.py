from ardis import *
import ardis.d_geometry as dg
import ardis.geometry as geo

import numpy as np

import os

matrixFolder = "data"
outputFolder = "output"

reaction, drain, epsilon = 1,  1e-8, 1e-3

dt, max_time = 0.1, 10

plot_dt = 1

name = "maze"

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

U = d_vector(n)
U.fill_value(0)

d_Mesh = dg.d_mesh(Mesh.x, Mesh.y)

startZone = dg.rect_zone(dg.point2d(0, 0), dg.point2d(500, 10))
dg.fill_zone(U, d_Mesh, startZone, 1)

st.add_species("N")
st.set_species("N", U)
st.add_species("P")
st.set_species("P", U)
st.add_species("NP")
st.set_species("NP", np.zeros(len(U)))

simu = simulation(st)
simu.drain = drain
simu.epsilon = epsilon

simu.load_stiffness_matrix(d_S)
simu.load_dampness_matrix(d_D)

simu.add_mm_reaction(" N -> 2 N", reaction, 1)
simu.add_reaction(" N+P -> NP", reaction)
simu.add_mm_reaction(" NP -> 2P", reaction, 1)
# simu.add_reaction("N+P-> 2P", reaction)

Nit = int(max_time / dt)
verboseCount = 0
plotcount = 0

os.system("mkdir " + outputFolder + "/" + name)

for i in range(0, Nit):
    simu.iterate_diffusion(dt)
    simu.prune()
    simu.iterate_reaction(dt, True)

    os.system("rm -f test")
    write_file(simu.state, "test")
    print(simu.state == read_state("test"))

    if (i * dt > plot_dt * plotcount):
        plotcount += 1
        fig = plot_state(simu.state, Mesh, excludeSpecies=["NP"])
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
