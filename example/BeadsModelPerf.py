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

dt, max_time = 0.01, 40
# plot_dt = 1


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

Nit = int(max_time / dt)

t0 = time.time()

for expr in range(0, 100):
    print("Starting experiment", expr)
    simu = simulation(d_D.shape[0])
    import_crn(simu, "chemicalReactionNetworks/t_approx.json")

    simu.drain = drain
    simu.epsilon = epsilon

    simu.load_stiffness_matrix(d_S)
    simu.load_dampness_matrix(d_D)

    dg.fill_zone(simu.get_species("0"), d_Mesh, upleftcorner, 1)
    dg.fill_zone(simu.get_species("1"), d_Mesh, uprightcorner, 1)

    for i in range(0, Nit):

        dg.fill_zone(simu.get_species("0"), d_Mesh, upleftcorner, 1)
        dg.fill_zone(simu.get_species("1"), d_Mesh, uprightcorner, 1)

        simu.iterate_reaction(dt)
        simu.iterate_diffusion(dt)

    print("total time until now:", time.time() - t0)

    # simu.print_profiler()
