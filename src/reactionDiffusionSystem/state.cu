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

D_Array **State::GetDeviceState() {
    D_Array **output = new D_Array *[data.size()];
    D_Array **d_output;
    cudaMalloc(&d_output, sizeof(D_Array *) * data.size());
    for (int i = 0; i < data.size(); i++) {
        output[i] = data.at(i)._device;
    }
    cudaMemcpy(d_output, output, sizeof(D_Array *) * data.size(),
               cudaMemcpyHostToDevice);
    delete output;
    return d_output;
}

void State::FreeDeviceState(D_Array **freeMe) { cudaFree(freeMe); }
