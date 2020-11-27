#pragma once

#include <map>
#include <vector>

#include "constants.hpp"
#include "dataStructures/array.hpp"
#include "dataStructures/sparse_matrix.hpp"

class State {
  public:
    // Size of the concentration vectors
    int vector_size;
    std::vector<D_Vector> vector_holder;
    D_Array<D_Vector *> device_data;
    // Stores the names of the species corresponding to each vector
    std::map<std::string, int> names;

    State(int size);

    // Adds a new species (doesn't allocate memory)
    D_Vector &AddSpecies(std::string name);
    D_Vector &GetSpecies(std::string name);

    // Uses the given vector as a concentration vector
    void SetSpecies(std::string name, D_Vector &sub_state);

    int size();
    int n_species();
    // Give as input the number of elements of each vector to be printed
    void Print(int i = 5);

    ~State();
};
