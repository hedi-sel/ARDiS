#pragma once

#include <map>
#include <vector>

#include "constants.hpp"
#include "dataStructures/array.hpp"
#include "dataStructures/sparse_matrix.hpp"

struct species_options {
    species_options(bool = true);
    bool diffusion;
};

class state {
  public:
    // Data holders
    int vector_size;
    std::vector<d_vector> vector_holder;
    std::vector<species_options> options_holder;
    // Stores the names of the species corresponding to each vector
    std::map<std::string, int> names;

    state(int size);
    state(state &&);
    state(const state &other);

    void operator=(const state &other);

    // Adds a new species (doesn't allocate memory)
    d_vector &add_species(std::string name,
                          species_options = species_options());
    d_vector &get_species(std::string name);

    // Uses the given vector as a concentration vector
    void set_species(std::string name, const T *data, bool is_device);
    void set_species(std::string name, d_vector &sub_state);

    d_array<d_vector *> &get_device_data();

    int size() const;
    int n_species() const;
    // Give as input the number of elements of each vector to be printed
    void print(int i = 5);

    ~state();

  private:
    d_array<d_vector *> device_data;
    void update_device_data();
};
