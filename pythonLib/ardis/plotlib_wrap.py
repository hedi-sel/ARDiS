import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as clr
import numpy as np
import math
from matplotlib.colors import ListedColormap

color_list = [
    'Reds', 'Blues', 'Greens', 'Purples', 'Oranges'
]


def plot_state(state, mesh, title="no title", listSpecies=[], excludeSpecies=[], colors={}):
    scatterSize = 3000 * (np.max(mesh.x) - np.min(mesh.x)) * \
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
    ax.scatter(mesh.x, mesh.y, s=2*scatterSize, marker='s',
                c=[background_color], vmin=0, vmax=1)
    for species in listSpecies:
        if species in excludeSpecies:
            continue
        vect = state.get_species(species).toarray()
        if type(colors[species]) == str:
            N = 100
            col_map = plt.get_cmap(colors[species], N)
            col_map = col_map(np.arange(N))
            col_map[:, -1] = np.linspace(0, 1, N)**2
            col_map = ListedColormap(col_map)
        else:
            col_map = ListedColormap(np.linspace(
               (0,0,0,0), clr.to_rgba(colors[species]), 20))
        ax.scatter(mesh.x, mesh.y, s=scatterSize, marker='o', c=vect, vmin=0, vmax=max(vect), cmap=col_map) 
    return fig
