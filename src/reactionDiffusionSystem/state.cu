#include <stdio.h>

#include "state.hpp"

State::State(int size) : size(size) {}

D_Array &State::AddSpecies(std::string name) {
    names[name] = data.size();
    data.push_back(size);
    return data.at(data.size() - 1);
}

D_Array &State::GetSpecies(std::string name) { return data.at(names[name]); }

void State::Print(int i) {
    for (auto name : names) {
        std::cout << name.first << " : ";
        data.at(name.second).Print(i);
    }
}

State::~State() {}
