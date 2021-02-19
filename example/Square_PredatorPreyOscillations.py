from ardis import *
import ardis.d_geometry as dg
import ardis.geometry as geo

import numpy as np

import os
#Set paths and locations
matrixFolder = "data"
outputFolder = "output"

dataName = "square"

dampingPath = matrixFolder+"/"+dataName+"_damping.mtx"
stiffnessPath = matrixFolder+"/"+dataName+"_stiffness.mtx"
meshPath = matrixFolder + "/" + dataName + "_mesh.dat"

#Load the mesh
Mesh = dg.read_mesh(meshPath)
#Copy the mesh to the GPU
d_Mesh = dg.d_mesh(Mesh.x, Mesh.y)

#Load Damping and stiffness matrices
S = read_spmatrix(stiffnessPath, read_type.Symetric)
D = read_spmatrix(dampingPath, read_type.Symetric)

#Copy the matrices to the GPU)
d_S = to_d_spmatrix(S, matrix_type.CSR)
d_D = to_d_spmatrix(D, matrix_type.CSR)

#Create a simulation
n = len(Mesh.x) #Number of nodes
simu = simulation(n)


d_Mesh = dg.d_mesh(Mesh.x, Mesh.y)


#Create 2 species N and P

simu.add_species("N")
simu.add_species("P")

#Species N will have concentration 1 in the upper-left corner and 0 everywhere else

N = d_vector(n)
N.fill_value(0)
upleftcorner = dg.rect_zone(dg.point2d(0, 7), dg.point2d(3, 10))
dg.fill_zone(N, d_Mesh, upleftcorner, 1)

simu.set_species("N", N)

#Species P will have concentration 1 in the upper-left corner and 0 everywhere else

P = d_vector(n)
P.fill_value(0)
bottomrightcorner = dg.rect_zone(dg.point2d(7, 0), dg.point2d(10, 3))
bottomleftcorner = dg.rect_zone(dg.point2d(0, 0), dg.point2d(3, 3))
dg.fill_zone(P, d_Mesh, bottomleftcorner, 1)

simu.set_species("P", P)

#Define chemical reactions
reaction = 1
decay = 0.1
simu.add_reaction(" N -> 2 N", reaction)
simu.add_reaction(" N+P -> 2P", reaction)
simu.add_reaction(" 2 N -> ", reaction)
simu.add_reaction(" P -> ", 0.5)

#Set all the necessary parameters
simu.load_stiffness_matrix(d_S)
simu.load_dampness_matrix(d_D)

Nit = 1000
dt = 0.1
plot_dt = 0.5
verboseCount = 0
plotcount = 0

name="test"
os.system("rm -r " + outputFolder + "/" + name)
os.system("mkdir " + outputFolder + "/" + name)

for i in range(0, Nit):
    simu.iterate_diffusion(dt/10)
    simu.prune()
    simu.iterate_reaction(dt)

    if (i * dt > plot_dt * plotcount):
        plotcount += 1
        fig = plot_state(simu.state, Mesh, colors={'background':(1,1,1,0)})
        fig.savefig(
            outputFolder + "/"+name+"/" + str(i) + ".png")
        plt.close(fig)

    if Nit >= 100 and i >= verboseCount * Nit / 10 and i < verboseCount * Nit / 10 + 1:
        print(str(verboseCount * 10) + "% completed")
        verboseCount += 1

os.system("convert -delay 10 -loop 0 $(ls -1 "+outputFolder +
          "/" + name + "/*png | sort -V) " + outputFolder + "/" + name + ".gif")

print("Results plot have been saved here: " +
      outputFolder + "/" + name + ".gif")
