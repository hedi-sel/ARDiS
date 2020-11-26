#include <stdio.h>

#include "state.hpp"

State::State(int size) : size(size) {}

D_Vector &State::AddSpecies(std::string name) {
    deviceStateToBeUpdated = true;
    mappingStateToBeUpdated = true;
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

int State::Size() { return size; }

void State::Print(int i) {
    for (auto name : names) {
        std::cout << name.first << " : ";
        data.at(name.second)->Print(i);
    }
}

State::~State() {
    for (auto arrPtr : data)
        delete arrPtr;
    if (deviceState != NULL)
        cudaFree(deviceState);
}

D_Vector **State::GetDeviceState() {
    if (deviceStateToBeUpdated) {
        D_Vector **output = new D_Vector *[data.size()];
        if (deviceState != NULL)
            cudaFree(deviceState);
        cudaMalloc(&deviceState, sizeof(D_Vector *) * data.size());
        for (int i = 0; i < data.size(); i++) {
            output[i] = (D_Vector *)data.at(i)->_device;
        }
        cudaMemcpy(deviceState, output, sizeof(D_Vector *) * data.size(),
                   cudaMemcpyHostToDevice);
        delete output;
        deviceStateToBeUpdated = false;
    }
    return deviceState;
}

__global__ void GetMappingStateK(D_Vector **deviceState, T ***mappingState,
                                 int nSpecies) {
    int n = deviceState[0]->Size();
    mappingState = new T **[n];
    for (int i = 0; i < n; i++) {
        mappingState[i] = new T *[nSpecies];
        for (int j = 0; j < nSpecies; j++) {
            mappingState[i][j] = &deviceState[j]->At(i);
        }
    }
}

T ***State::GetMappingState() {
    if (mappingStateToBeUpdated) {
        if (deviceState != nullptr)
            cudaFree(deviceState);
        GetMappingStateK<<<1, 1>>>(GetDeviceState(), mappingState, data.size());
        mappingStateToBeUpdated = false;
    }
    return mappingState;
}