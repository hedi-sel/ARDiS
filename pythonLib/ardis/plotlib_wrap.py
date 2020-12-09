import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as colors
import numpy as np
import math

color_list = [
    [[0.8, 0.1, 0.2]],
    [[0.2, 0.4, 0.6]],
    [[0.2, 0.6, 0.2]],
    [[0.8, 0.8, 0.8]]
]


def plot_state(state, mesh, title="no title", listSpecies=[], excludeSpecies=[], colors={}):
    scatterSize = math.sqrt((np.max(mesh.x) - np.min(mesh.x))
                            * (np.max(mesh.y) - np.min(mesh.y)) * 1.0 / state.vector_size())
    if(len(listSpecies) == 0):
        listSpecies = state.list_species()

    col_count = 0
    for species in listSpecies:
        if species in excludeSpecies:
            continue
        if species not in colors:
            colors[species] = color_list[col_count]
            col_count = (col_count+1) % len(color_list)
    if "background" in colors:
        background_color = colors["background"]
    else:
        background_color = [[0, 0.1, 0.3]]

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.scatter(mesh.x, mesh.y, s=4*scatterSize,
                c=[[0, 0.1, 0.3]], vmin=0, vmax=1)
    for species in listSpecies:
        if species in excludeSpecies:
            continue
        vect = state.get_species(species).toarray()
        ax.scatter(mesh.x, mesh.y, s=vect *
                   scatterSize, vmin=0, vmax=1, c=colors[species])
    return fig
