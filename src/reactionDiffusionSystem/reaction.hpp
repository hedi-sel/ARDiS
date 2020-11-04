#pragma once

// #include <nvfunctional>
#include <string>
#include <vector>

#include "constants.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "matrixOperations/basic_operations.hpp"

typedef std::pair<std::string, int> stochCoeff;

// class Rate {
//     __device__ void Function(const T &reagent, T &progress);
// };

class Reaction {
  public:
    std::vector<stochCoeff> Reagents;
    std::vector<stochCoeff> Products;

    // nvstd::function<void(void)> *Rate = &([] __device__() { printf("Hi\n");
    // }); nvstd::function<void(void)> Rate = [] __device__() { printf("Hi\n");
    // };

    Reaction(std::vector<stochCoeff>, std::vector<stochCoeff>);
};

class ReactionMassAction : public Reaction {
  public:
    T K;

    ReactionMassAction(Reaction, T);
    ReactionMassAction(std::vector<stochCoeff>, std::vector<stochCoeff>, T);

    __device__ void Rate(const T &reagent, T &progress);
};

class ReactionMichaelisMenten : public Reaction {
  public:
    T Vm;
    T Km;

    ReactionMichaelisMenten(Reaction, T, T);
    ReactionMichaelisMenten(std::string, std::vector<stochCoeff>, T, T);

    __device__ void Rate(const T &reagent, T &progress);
};
