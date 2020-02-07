#pragma once

#include <map>
#include <vector>

#include "constants.hpp"
#include "dataStructures/array.hpp"
#include "dataStructures/sparse_matrix.hpp"

class State {
  public:
    int size;
    std::vector<D_Array> data;
    std::map<std::string, int> names;

    State(int size);

    D_Array &AddSpecies(std::string name);

    D_Array &GetSpecies(std::string name);

    void SetSpecies(std::string name, D_Array &sub_state);

    void Print(int i = 5);

    ~State();

    // Device related functions
    D_Array **GetDeviceState();
    void FreeDeviceState(D_Array **);
};