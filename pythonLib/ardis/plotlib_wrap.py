import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as colors
import numpy as np
import math
from matplotlib.colors import ListedColormap

color_list = [
    (0.8, 0.1, 0.2, 1),
    (0.2, 0.4, 0.6, 1),
    (0.2, 0.6, 0.2, 1),
    (0.8, 0.8, 0.8, 1)
]


def plot_state(state, mesh, title="no title", listSpecies=[], excludeSpecies=[], colors={}):
    scatterSize = 4000 * (np.max(mesh.x) - np.min(mesh.x)) * \
        (np.max(mesh.y) - np.min(mesh.y)) * 1.0 / state.vector_size()
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
        background_color = (0, 0.1, 0.3)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    # plt.tripcolor(mesh, np.ones(mesh.x.size),
    #               cmap=background_color, vmin=0, vmax=1)
    plt.scatter(mesh.x, mesh.y, s=2*scatterSize, marker='o',
                c=[background_color], vmin=0, vmax=1)
    for species in listSpecies:
        if species in excludeSpecies:
            continue
        vect = state.get_species(species).toarray()
        col_map = ListedColormap(np.linspace(
            (0, 0, 0, 0), colors[species], 20))
        ax.scatter(mesh.x, mesh.y, s=scatterSize, marker='o',
                   c=vect, vmin=0, vmax=1, cmap=col_map)  # vect
    return fig
