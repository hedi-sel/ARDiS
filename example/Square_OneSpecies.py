from ardis import *
import ardis.d_geometry as dg
import ardis.geometry as geo
import numpy as np
import os


#Set paths and locations
matrixFolder = "data"
outputFolder = "output"

dataName = "coarse_square"

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

#Set parameters
simu.drain = 0
simu.epsilon = 1.e-3

#Load matrices into the simulation
simu.load_stiffness_matrix(d_S)
simu.load_dampness_matrix(d_D)

#Create a species and set its concentration to zero
simu.add_species("A")
simu.set_species("A", np.zeros(n))

#Set the concentration of species A to 10 in the upper-left corner
upleftcorner = dg.rect_zone(dg.point2d(0, 7), dg.point2d(3, 10))
dg.fill_zone(simu.get_species("A"), d_Mesh, upleftcorner, 10)

#Plot the initial state of the simulation
fig = plot_state(simu.state, Mesh, title="square_onespecies", listSpecies=["A"], colors={"background": (0,0,0,0), "A": (0.8, 0.2, 0, 1)})
plt.show()
fig.savefig(outputFolder +"/square_onespecies_t=0.png")
plt.close(fig)

#Iterate diffusion
dt = 1
for i in range(0, 10):
    simu.iterate_diffusion(dt)

#Plot the final state of the simulation
fig = plot_state(simu.state, Mesh, title="square_onespecies", listSpecies=["A"], colors={"background": (0,0,0,0), "A": (0.8, 0.2, 0, 1)})
plt.show()
fig.savefig(outputFolder +"/square_onespecies_t=10.png")
plt.close(fig)

print("Simulation completed!")
print("Figures saved in '" + outputFolder + "'") 
