#include <stdio.h>

#include "helper/cuda/cuda_error_check.h"
#include "state.hpp"

state::state(int size) : vector_size(size) {}
state::state(state &&other)
    : vector_size(other.size()), vector_holder(std::move(other.vector_holder)),
      names(std::move(other.names)) {}
void state::operator=(const state &other) {
    names = other.names;
    vector_size = other.vector_size;
    vector_holder = other.vector_holder;
}

void state::update_device_data() {
    device_data.resize(n_species());
    d_vector *new_device_data[n_species()];
    for (int i = 0; i < n_species(); i++)
        new_device_data[i] = (d_vector *)vector_holder.at(i)._device;
    gpuErrchk(cudaMemcpy(device_data.data, new_device_data,
                         sizeof(d_vector *) * n_species(),
                         cudaMemcpyHostToDevice));
}

d_array<d_vector *> &state::get_device_data() {
    if (device_data.size() < n_species()) {
        update_device_data();
    }
    return device_data;
}

d_vector &state::add_species(std::string name) {
    names[name] = n_species();
    vector_holder.push_back(d_vector(vector_size));
    return vector_holder.at(n_species() - 1);
}

d_vector &state::get_species(std::string name) {
    auto findRes = names.find(name);
    if (findRes == names.end()) {
        std::cout << "\"" << name << "\"\n";
        throw std::invalid_argument("^ This species is invalid\n");
    }
    return vector_holder.at(names[name]);
}

void state::set_species(std::string name, const T *data, bool is_device) {
    gpuErrchk(cudaMemcpy(get_species(name).data, data, sizeof(T) * size(),
                         (is_device) ? cudaMemcpyDeviceToDevice
                                     : cudaMemcpyHostToDevice));
}

int state::size() { return vector_size; }
int state::n_species() { return vector_holder.size(); }

void state::print(int i) {
    for (auto name : names) {
        std::cout << name.first << " : ";
        vector_holder.at(name.second).print(i);
    }
}

state::~state() {}
