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
    std::vector<D_Vector *> data;
    // Stores the names of the species corresponding to each vector
    std::map<std::string, int> names;

    State(int size);

    // Adds a new species (doesn't allocate memory)
    D_Vector &AddSpecies(std::string name);
    D_Vector &GetSpecies(std::string name);

    // Uses the given vector as a concentration vector
    void SetSpecies(std::string name, D_Vector &sub_state);

    // Give as input the number of elements of each vector to be printed
    void Print(int i = 5);

    ~State();

    // Function used to get all the data as a device friendly pointer
    bool deviceStateToBeUpdated = true;
    D_Vector **deviceState = NULL;
    D_Vector **GetDeviceState();

    // Function used to transpose the date (so we can access teh concentrations
    // of all species at one specific point)
    bool mappingStateToBeUpdated = true;
    T ***mappingState = nullptr;
    T ***GetMappingState();
};