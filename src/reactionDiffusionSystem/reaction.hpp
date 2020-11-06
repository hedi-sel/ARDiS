#pragma once

// #include <nvfunctional>
#include <string>
#include <vector>

#include "constants.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "matrixOperations/basic_operations.hpp"

typedef std::pair<std::string, int> stochCoeff;

class Reaction {
  public:
    std::vector<stochCoeff> Reagents;
    std::vector<stochCoeff> Products;

    Reaction(std::vector<stochCoeff>, std::vector<stochCoeff>);

    T BaseRate(T dt);

    void Print();
};

class ReactionMassAction : public Reaction {
  public:
    T K;

    ReactionMassAction *_device;

    ReactionMassAction(Reaction, T);
    ReactionMassAction(std::vector<stochCoeff>, std::vector<stochCoeff>, T);

    T BaseRate(T dt);
    __device__ void Rate(const T &reagent, T &progress);

    void Print();
};

class ReactionMichaelisMenten : public Reaction {
  public:
    T Vm;
    T Km;

    ReactionMichaelisMenten *_device;

    ReactionMichaelisMenten(Reaction, T, T);
    ReactionMichaelisMenten(std::string, std::vector<stochCoeff>, T, T);

    T BaseRate(T dt);
    __device__ void Rate(const T &reagent, T &progress);

    void Print();
};
