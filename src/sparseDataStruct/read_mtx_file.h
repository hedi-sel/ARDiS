#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "sparseDataStruct/matrix_sparse.hpp"

__host__ MatrixSparse *ReadFromFile(const std::string filepath,
                                    MatrixSparse *matrix = nullptr) {
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

    matrix = new MatrixSparse(i, j, n_elts, COO);

    for (int k = 0; k < n_elts; k++) {
        while (std::getline(myfile, line)) {
            if (line[0] != '%') {
                std::istringstream iss(line);
                if (!(iss >> i >> j >> val))
                    std::cerr << "Error line: " << line << std::endl;
                else {
                    i--;
                    j--;
                    assert(i >= 0 && i < matrix->i_size && j >= 0 &&
                           j < matrix->j_size);
                    matrix->AddElement(k, i, j, val);
                    break;
                }
            }
        }
    }
    myfile.close();
    return matrix;
}