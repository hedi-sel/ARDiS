#pragma once

#include <map>
#include <vector>

#include "constants.hpp"
#include "dataStructures/array.hpp"
#include "dataStructures/sparse_matrix.hpp"

class state {
  public:
    // Size of the concentration vectors
    int vector_size;
    std::vector<d_vector> vector_holder;
    d_array<d_vector *> device_data;
    // Stores the names of the species corresponding to each vector
    std::map<std::string, int> names;

    state(int size);

    // Adds a new species (doesn't allocate memory)
    d_vector &add_species(std::string name);
    d_vector &get_species(std::string name);

    // Uses the given vector as a concentration vector
    void set_species(std::string name, d_vector &sub_state);

    int size();
    int n_species();
    // Give as input the number of elements of each vector to be printed
    void print(int i = 5);

    ~state();
};
