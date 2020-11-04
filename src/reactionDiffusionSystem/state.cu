#include <stdio.h>

#include "state.hpp"

State::State(int size) : size(size) {}

D_Vector &State::AddSpecies(std::string name) {
    names[name] = data.size();
    data.push_back(new D_Vector(size));
    return *(data.at(data.size() - 1));
}

D_Vector &State::GetSpecies(std::string name) {
    auto findRes = names.find(name);
    if (findRes == names.end()) {
        std::cout << "\"" << name << "\""
                  << "\n";
        throw std::invalid_argument("^ This species is invalid\n");
    }
    return *(data.at(names[name]));
}

void State::Print(int i) {
    for (auto name : names) {
        std::cout << name.first << " : ";
        data.at(name.second)->Print(i);
    }
}

State::~State() {
    for (auto arrPtr : data)
        delete arrPtr;
}

D_Vector **State::GetDeviceState() {
    D_Vector **output = new D_Vector *[data.size()];
    D_Vector **d_output;
    cudaMalloc(&d_output, sizeof(D_Vector *) * data.size());
    for (int i = 0; i < data.size(); i++) {
        output[i] = (D_Vector *)data.at(i)->_device;
    }
    cudaMemcpy(d_output, output, sizeof(D_Vector *) * data.size(),
               cudaMemcpyHostToDevice);
    delete output;
    return d_output;
}

void State::FreeDeviceState(D_Vector **freeMe) { cudaFree(freeMe); }
