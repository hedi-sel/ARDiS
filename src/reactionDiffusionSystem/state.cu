#include <stdio.h>

#include "helper/cuda/cuda_error_check.h"
#include "state.hpp"
State::State(int size) : vector_size(size) {}

D_Vector &State::AddSpecies(std::string name) {
    names[name] = n_species();
    vector_holder.push_back(D_Vector(vector_size));
    device_data.Resize(n_species());
    D_Vector *new_device_data[n_species()];
    for (int i = 0; i < n_species(); i++)
        new_device_data[i] = (D_Vector *)vector_holder.at(i)._device;
    gpuErrchk(cudaMemcpy(device_data.data, new_device_data,
                         sizeof(D_Vector *) * n_species(),
                         cudaMemcpyHostToDevice));
    return vector_holder.at(n_species() - 1);
}

D_Vector &State::GetSpecies(std::string name) {
    auto findRes = names.find(name);
    if (findRes == names.end()) {
        std::cout << "\"" << name << "\"\n";
        throw std::invalid_argument("^ This species is invalid\n");
    }
    return vector_holder.at(names[name]);
}

int State::size() { return vector_size; }
int State::n_species() { return vector_holder.size(); }

void State::Print(int i) {
    for (auto name : names) {
        std::cout << name.first << " : ";
        vector_holder.at(name.second).Print(i);
    }
}

State::~State() {}
