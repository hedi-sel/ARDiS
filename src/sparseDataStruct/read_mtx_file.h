#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "sparseDataStruct/matrix_sparse.hpp"

enum Readtype { Normal, Symetric };

Readtype readtype = Symetric;
__host__ MatrixSparse ReadFromFile(const std::string filepath/* ,
                                   Readtype readtype = Normal */) {
    int i, j;
    int n_elts = 0;
    T val;

    std::string line;
    std::ifstream myfile;
    myfile.open(filepath);
    if (!myfile.is_open()) {
        printf("File could not be opened");
    }
    while (n_elts == 0 && std::getline(myfile, line)) {
        if (line[0] != '%') {
            std::istringstream iss(line);
            if (!(iss >> i >> j >> n_elts))
                std::cerr << "Error line\n";
        }
    }

    MatrixSparse matrix(i, j, (readtype == Normal) ? n_elts : n_elts * 2 - i,
                        COO);

    for (int k = 0; k < n_elts; k++) {
        do {
            std::getline(myfile, line);
        } while (line[0] == '%');

        std::istringstream iss(line);
        if (!(iss >> i >> j >> val))
            std::cerr << "Error line: " << line << std::endl;
        else {
            i--;
            j--;
            assert(i >= 0 && i < matrix.i_size && j >= 0 && j < matrix.j_size);
            matrix.AddElement(k, i, j, val);
            if (i != j)
                matrix.AddElement(k, j, i, val);
        }
    }
    assert(!std::getline(myfile,
                         line)); // We check that there is no more lines left to
                                 // be read after the matrix is complete
    myfile.close();
    return matrix;
}