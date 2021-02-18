from .ardisLib import state

import numpy as np
from scipy.sparse import *
import matplotlib.tri as tri
from enum import Enum
import json


def line_to_values(line):
    if (line[0] == '%'):
        return None
    values = []
    splitString = line.split(" ")
    if(len(splitString) == 1):
        splitString = line.split("\t")
    for str in splitString:
        if str == "":
            continue
        try:
            values.append(int(str))
        except ValueError:
            try:
                values.append(float(str))
            except ValueError:
                values.append(str.replace("\n", ""))
    return values


class read_type(Enum):
    Normal = 0
    Symetric = 1


def read_spmatrix(path, readtype=read_type.Normal):
    f = open(path, "r")
    lines = f.readlines()
    i = -1
    line = None
    while (line == None):
        line = line_to_values(lines.pop(0))
    i, j, k = line
    mat = lil_matrix((i, j))
    for line in lines:
        values = None
        while (values == None):
            values = line_to_values(line)
        if len(values) == 3:
            mat[values[0] - 1, values[1] - 1] = values[2]
            if (readtype == read_type.Symetric and values[1] != values[0]):
                mat[values[1] - 1, values[0] - 1] = values[2]
        else:
            print(values)
            print("Could not read the following line: ", line)
    return mat


def read_state(path):

    f = open(path)
    lines = f.readlines()

    vect_size, n_species = line_to_values(lines.pop(0))

    imp_state = state(vect_size)
    species_list = {}

    for i in range(0, n_species):
        species_idx, species_name = line_to_values(lines.pop(0))
        species_list[species_idx] = species_name

    for i in range(0, n_species):
        imp_state.add_species(str(species_list[i]))

    for i in range(0, n_species):
        vect = lines.pop(0).split("\t")
        species = vect.pop(0)
        vect = np.array(vect)
        imp_state.get_species(species).import_array(vect)
    return imp_state


def import_crn(simu, path):
    data = json.load(open(path))
    simu.add_species("trash", diffusion=False)
    inhibitors = []
    for sp in data['nodes']:
        simu.add_species(sp['name'])
        simu.set_species(sp['name'], np.ones(len(simu.state))*1.e-10)

        if (sp['name'][0] == 'I' and 'T' in sp['name']):
            inhibitors.append(sp['name'])
            simu.add_mm_reaction(sp['name'] + "-> " + "trash", 300, 150)
        else:
            simu.add_mm_reaction(sp['name'] + "-> " + "trash", 300, 440)

    for reac in data['connections']:
        sp_from = reac['from']
        sp_to = reac['to']
        template = "template_" + sp_from + "-" + sp_to
        template_bind_from = template + "~" + sp_from
        template_bind_to = template+"~" + sp_to
        template_bind_from_to = template + "~"+sp_from+"|"+sp_to
        template_bind_fromto = template + "~" + sp_from + "-" + sp_to

        simu.add_species(template, diffusion=False)
        simu.add_species(template_bind_from, diffusion=False)
        simu.add_species(template_bind_to, diffusion=False)
        simu.add_species(template_bind_from_to, diffusion=False)
        simu.add_species(template_bind_fromto, diffusion=False)

        simu.set_species(template, np.ones(
            len(simu.state))*1e-2*reac['parameter'])
        simu.set_species(template_bind_from, np.zeros(len(simu.state)))
        simu.set_species(template_bind_to, np.zeros(len(simu.state)))
        simu.set_species(template_bind_from_to, np.zeros(len(simu.state)))
        simu.set_species(template_bind_fromto, np.zeros(len(simu.state)))

        simu.add_reversible_reaction(
            template+"+"+sp_from+"->"+template_bind_from, 0.2, 0.2)
        simu.add_reversible_reaction(
            template+"+"+sp_to+"->"+template_bind_to, 0.2, 0.2)
        simu.add_reversible_reaction(
            template_bind_to+"+"+sp_from+"->"+template_bind_from_to, 0.2, 0.2)
        simu.add_reversible_reaction(
            template_bind_from+"+"+sp_to+"->"+template_bind_from_to, 0.2, 0.2)
        simu.add_mm_reaction(
            template_bind_from+" -> "+template_bind_fromto, 1050, 80)
        simu.add_mm_reaction(
            template_bind_from_to+" -> "+template_bind_fromto + "+" + sp_to, 1050, 80)
        simu.add_mm_reaction(
            template_bind_fromto+" -> "+template_bind_from_to, 80, 30)

        if "I"+sp_from+"T"+sp_to in inhibitors:
            inhib = "I"+sp_from+"T"+sp_to
            template_inhibited = template + "~" + inhib
            simu.add_species(template_inhibited, diffusion=False)
            simu.add_reaction(template+"+"+inhib+"->"+template_inhibited, 0.2)
            simu.add_reaction(template_bind_to+"+"+inhib +
                              "->"+template_inhibited+" + "+sp_to, 0.2)
            simu.add_reaction(template_bind_from+"+"+inhib +
                              "->"+template_inhibited+" + "+sp_from, 0.2)
