import ardis as ar

print("Ardis library succesfully loaded")

matrixFolder = "data"
dampingPath = matrixFolder+"/coarse_square_damping.mtx"
stiffnessPath = matrixFolder+"/coarse_square_stiffness.mtx"
meshPath = matrixFolder + "/coarse_square_mesh.dat"


try:
    S = ar.read_spmatrix(stiffnessPath, ar.read_type.Symetric)
    d_S = ar.to_d_spmatrix(S , ar.matrix_type.CSR)
    print("Example data succesfully loaded")

except:
    print("Example data could not be loaded. Did you unpack 'data.zip'?")


try:
    import ardis.geometry as geo
    import ardis.d_geometry as dg

    Mesh = dg.read_mesh(meshPath)

    print("Geometry library succesfully loaded")
except:
    print("Geometry library could not be loaded")
