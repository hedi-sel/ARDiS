#pragma once

// #include <nvfunctional>
#include <string>
#include <vector>

#include "constants.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "matrixOperations/basic_operations.hpp"

class state;

typedef std::pair<std::string, int> stochCoeff;

struct reaction_holder {
    reaction_holder(std::vector<stochCoeff>, std::vector<stochCoeff>);
    std::vector<stochCoeff> Reagents;
    std::vector<stochCoeff> Products;
};

class reaction {
  public:
    reaction_holder Holder;

    d_array<int> Reagents;
    d_array<int> ReagentsCoeff;
    d_array<int> Products;
    d_array<int> ProductsCoeff;
    d_array<int> Inhibitor;

    // reaction *_device;

    reaction(std::map<std::string, int> &names, std::vector<stochCoeff>,
             std::vector<stochCoeff>, long unsigned size);
    reaction(std::map<std::string, int> &names, reaction_holder,
             long unsigned size);
    reaction(reaction &&);

    void add_inhibitor(int);

    __host__ __device__ void print() const;

    ~reaction();
};

class reaction_mass_action : public reaction {
  public:
    T K;

    reaction_mass_action *_device;

    reaction_mass_action(std::map<std::string, int> &names, reaction_holder, T);
    reaction_mass_action(std::map<std::string, int> &names,
                         std::vector<stochCoeff>, std::vector<stochCoeff>, T);

    inline __device__ void ApplyReaction(d_array<d_vector *> &state, int i,
                                         float dt) {
        T progress = K * dt;
        if (Inhibitor.at(0) != -1)
            progress *= 1 / (1 + state.at(Inhibitor.at(0))->at(i));

        for (int k = 0; k < Reagents.size(); k++)
            progress *=
                pow(state.at(Reagents.at(k))->at(i), ReagentsCoeff.at(k));
        for (int k = 0; k < Reagents.size(); k++)
            state.at(Reagents.at(k))->at(i) -= progress * ReagentsCoeff.at(k);
        for (int k = 0; k < Products.size(); k++)
            state.at(Products.at(k))->at(i) += progress * ProductsCoeff.at(k);
    }

    __host__ __device__ void print() const;
};

class reaction_michaelis_menten : public reaction {
  public:
    T Vm;
    T Km;

    reaction_michaelis_menten *_device;

    reaction_michaelis_menten(std::map<std::string, int> &names,
                              reaction_holder, T, T);
    reaction_michaelis_menten(std::map<std::string, int> &names, std::string,
                              std::vector<stochCoeff>, T, T);
    __device__ void inline ApplyReaction(d_array<d_vector *> &state, int i,
                                         float dt) {
        T progress = Km * dt / (Vm + state.at(Reagents.at(0))->at(i));
        auto &val = state.at(Reagents.at(0))->at(i);
        if (Inhibitor.at(0) != -1)
            progress *= val / (val + state.at(Inhibitor.at(0))->at(i));

        progress *= val;
        val -= progress;
        for (int k = 0; k < Products.size(); k++)
            state.at(Products.at(k))->at(i) += progress * ProductsCoeff.at(k);
        return;
    }

    __host__ __device__ void print() const;
};
