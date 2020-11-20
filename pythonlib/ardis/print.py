import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as colors

from .geometry import *

def Plot(state, mesh, title = "", listSpecies = []):
    scatterSize = math.sqrt( (np.max(mesh.x) - np.min(mesh.x)) * (np.max(mesh.y) - np.min(mesh.y)) *1.0 / state.size)
    if(len(listSpecies)):
        listSpecies = state.ListSpecies()

    ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.scatter(mesh.x, mesh.y, s=5 * scatterSize, c=[[0,0.1,0.3]], vmin=0, vmax=1)
    for species in listSpecies:
        vect = state.GetSpecies(species).ToNumpyArray()
        plt.scatter(mesh.x, mesh.y, s=2 * vect * scatterSize,  vmin=0, vmax=1)
    return plt