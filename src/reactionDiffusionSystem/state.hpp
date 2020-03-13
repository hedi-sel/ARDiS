#pragma once

#include <map>
#include <vector>

#include "constants.hpp"
#include "dataStructures/array.hpp"
#include "dataStructures/sparse_matrix.hpp"

class State {
  public:
    // Size of the concentration vectors
    int size;
    std::vector<D_Array *> data;
    // Stores the names of the species corresponding to each vector
    std::map<std::string, int> names;

    State(int size);

    // Adds a new species (doesn't allocate memory)
    D_Array &AddSpecies(std::string name);
    D_Array &GetSpecies(std::string name);

    // Uses the given vector as a concentration vector
    void SetSpecies(std::string name, D_Array &sub_state);

    // Give as input the number of elements of each vector to be printed
    void Print(int i = 5);

    ~State();

    // Function used to get all the data as a device friendly pointer
    D_Array **GetDeviceState();
    void FreeDeviceState(D_Array **);
};