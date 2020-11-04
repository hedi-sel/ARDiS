#include "reaction.hpp"

Reaction::Reaction(std::vector<stochCoeff> reag, std::vector<stochCoeff> prod)
    : Reagents(reag), Products(prod) {}

ReactionMassAction::ReactionMassAction(std::vector<stochCoeff> reag,
                                       std::vector<stochCoeff> prod, T rate)
    : Reaction(reag, prod), K(rate) {}

ReactionMassAction::ReactionMassAction(Reaction reac, T rate)
    : Reaction(reac), K(rate) {}

ReactionMichaelisMenten::ReactionMichaelisMenten(Reaction reac, T Vm, T Km)
    : Reaction(reac), Vm(Vm), Km(Km) {}

ReactionMichaelisMenten::ReactionMichaelisMenten(std::string reag,
                                                 std::vector<stochCoeff> prod,
                                                 T Vm, T Km)
    : Reaction(std::vector<stochCoeff>{stochCoeff(reag, 1)}, prod), Vm(Vm),
      Km(Km) {}

__device__ void ReactionMassAction::Rate(const T &reagent, T &progress) {
    progress *= reagent;
}

__device__ void ReactionMichaelisMenten::Rate(const T &reagent, T &progress) {}