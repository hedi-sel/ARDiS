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

    D_Array<int> *D_Reagents;
    D_Array<int> *D_ReagentsCoeff;
    D_Array<int> *D_Products;
    D_Array<int> *D_ProductsCoeff;

    Reaction(std::map<std::string, int> names, std::vector<stochCoeff>,
             std::vector<stochCoeff>);
    Reaction(std::map<std::string, int> names, ReactionHolder);
    T BaseRate(T dt);

    void Print();
};

class ReactionMassAction : public Reaction {
  public:
    T K;

    ReactionMassAction *_device;

    ReactionMassAction(std::map<std::string, int> names, ReactionHolder, T);
    ReactionMassAction(std::map<std::string, int> names,
                       std::vector<stochCoeff>, std::vector<stochCoeff>, T);

    T BaseRate(T dt);
    __device__ void Rate(const T &reagent, T &progress);

    void Print();
};

class ReactionMichaelisMenten : public Reaction {
  public:
    T Vm;
    T Km;

    ReactionMichaelisMenten *_device;

    ReactionMichaelisMenten(std::map<std::string, int> names, ReactionHolder, T,
                            T);
    ReactionMichaelisMenten(std::map<std::string, int> names, std::string,
                            std::vector<stochCoeff>, T, T);

    T BaseRate(T dt);
    __device__ void Rate(const T &reagent, T &progress);

    void Print();
};
