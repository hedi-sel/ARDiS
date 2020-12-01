#pragma once

// #include <nvfunctional>
#include <string>
#include <vector>

#include "constants.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "matrixOperations/basic_operations.hpp"

class State;

typedef std::pair<std::string, int> stochCoeff;

struct ReactionHolder {
    ReactionHolder(std::vector<stochCoeff>, std::vector<stochCoeff>);
    std::vector<stochCoeff> Reagents;
    std::vector<stochCoeff> Products;
};

class Reaction {
  public:
    ReactionHolder holder;

    D_Array<int> Reagents;
    D_Array<int> ReagentsCoeff;
    D_Array<int> Products;
    D_Array<int> ProductsCoeff;

    // Reaction *_device;

    Reaction(std::map<std::string, int> &names, std::vector<stochCoeff>,
             std::vector<stochCoeff>, long unsigned size);
    Reaction(std::map<std::string, int> &names, ReactionHolder,
             long unsigned size);
    Reaction(Reaction &&);

    __host__ __device__ void Print() const;

    ~Reaction();
};

class ReactionMassAction : public Reaction {
  public:
    T K;

    ReactionMassAction *_device;

    ReactionMassAction(std::map<std::string, int> &names, ReactionHolder, T);
    ReactionMassAction(std::map<std::string, int> &names,
                       std::vector<stochCoeff>, std::vector<stochCoeff>, T);

    inline __device__ void ApplyReaction(D_Array<D_Vector *> &state, int i,
                                         float dt) {
        T progress = K * dt;
        for (int k = 0; k < Reagents.size(); k++)
            progress *=
                pow(state.at(Reagents.at(k))->at(i), ReagentsCoeff.at(k));
        for (int k = 0; k < Reagents.size(); k++)
            state.at(Reagents.at(k))->at(i) -= progress * ReagentsCoeff.at(k);
        for (int k = 0; k < Products.size(); k++)
            state.at(Products.at(k))->at(i) += progress * ProductsCoeff.at(k);
    }

    __host__ __device__ void Print() const;
};

class ReactionMichaelisMenten : public Reaction {
  public:
    T Vm;
    T Km;

    ReactionMichaelisMenten *_device;

    ReactionMichaelisMenten(std::map<std::string, int> &names, ReactionHolder,
                            T, T);
    ReactionMichaelisMenten(std::map<std::string, int> &names, std::string,
                            std::vector<stochCoeff>, T, T);
    __device__ void inline ApplyReaction(D_Array<D_Vector *> &state, int i,
                                         float dt) {
        T progress = Km * dt / (Vm + state.at(Reagents.at(0))->at(i));
        for (int k = 0; k < Reagents.size(); k++)
            progress *=
                pow(state.at(Reagents.at(k))->at(i), ReagentsCoeff.at(k));
        for (int k = 0; k < Reagents.size(); k++)
            state.at(Reagents.at(k))->at(i) -= progress * ReagentsCoeff.at(k);
        for (int k = 0; k < Products.size(); k++)
            state.at(Products.at(k))->at(i) += progress * ProductsCoeff.at(k);
        return;
    }

    __host__ __device__ void Print() const;
};
