#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <fstream>
#include <math.h>
#include <stdio.h>

#include "dataStructures/array.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "matrixOperations/basic_operations.hpp"
#include "reactionDiffusionSystem/simulation.hpp"

void write_file(d_vector &array, std::string outputPath,
                std::string prefix = "", std::string suffix = "\n") {
    if (array.is_device) { // If device memory, copy to host, and restart the
                           // function
        d_vector h_copy(array, true);
        write_file(h_copy, outputPath, prefix, suffix);
        return;
    }
    // once we made sure the vector's data is on the host memory
    std::ofstream fout;
    fout.open(outputPath, std::ios_base::app);
    if (prefix != std::string("")) {
        fout << prefix << "\t";
    }
    for (size_t j = 0; j < array.n; j++)
        fout << ((j == 0) ? "" : "\t") << array.data[j];
    if (suffix != std::string(""))
        fout << suffix;
    fout.close();
}

void write_file(state &state, std::string outputPath) {
    std::ofstream fout;
    fout.open(outputPath, std::ios_base::app);
    fout << state.vector_size << "\t" << state.n_species() << "\n";
    for (auto sp : state.names) {
        fout << sp.second << "\t" << sp.first << "\n";
    }
    fout << state.vector_size << "\t" << state.n_species() << "\n";
    fout.close();
    for (auto sp : state.names) {
        write_file(state.vector_holder.at(sp.second), outputPath, sp.first,
                   "\n");
    }
}