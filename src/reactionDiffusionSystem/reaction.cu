#include "reaction.hpp"

ReactionHolder::ReactionHolder(std::vector<stochCoeff> reag,
                               std::vector<stochCoeff> prod)
    : Reagents(reag), Products(prod) {}

Reaction::Reaction(std::map<std::string, int> names, ReactionHolder holder)
    : holder(holder) {}

Reaction::Reaction(std::map<std::string, int> names,
                   std::vector<stochCoeff> reag, std::vector<stochCoeff> prod)
    : Reaction(names, ReactionHolder(reag, prod)) {}

void Reaction::Print() {
    for (auto coeff : holder.Reagents)
        std::cout << coeff.second << "." << coeff.first << " + ";
    std::cout << "-> ";
    for (auto coeff : holder.Products)
        std::cout << coeff.second << "." << coeff.first << " + ";
    std::cout << "\n";
}

/////// Mass Action

ReactionMassAction::ReactionMassAction(std::map<std::string, int> names,
                                       std::vector<stochCoeff> reag,
                                       std::vector<stochCoeff> prod, T rate)
    : ReactionMassAction(names, ReactionHolder(reag, prod), rate) {}

ReactionMassAction::ReactionMassAction(std::map<std::string, int> names,
                                       ReactionHolder reac, T rate)
    : Reaction(names, reac), K(rate) {
    gpuErrchk(cudaMalloc(&_device, sizeof(ReactionMassAction)));
    gpuErrchk(cudaMemcpy(_device, this, sizeof(ReactionMassAction),
                         cudaMemcpyHostToDevice));
}

T ReactionMassAction::BaseRate(T dt) { return K * dt; }

__device__ void ReactionMassAction::Rate(const T &reagent, T &progress) {
    progress *= reagent;
}

void ReactionMassAction::Print() {
    Reaction::Print();
    std::cout << "k=" << K << "\n";
}

/////// Michaleis Menten

ReactionMichaelisMenten::ReactionMichaelisMenten(
    std::map<std::string, int> names, ReactionHolder reac, T Vm, T Km)
    : Reaction(names, reac), Vm(Vm), Km(Km) {
    gpuErrchk(cudaMalloc(&_device, sizeof(ReactionMichaelisMenten)));
    gpuErrchk(cudaMemcpy(_device, this, sizeof(ReactionMichaelisMenten),
                         cudaMemcpyHostToDevice));
}

ReactionMichaelisMenten::ReactionMichaelisMenten(
    std::map<std::string, int> names, std::string reag,
    std::vector<stochCoeff> prod, T Vm, T Km)
    : ReactionMichaelisMenten(
          names,
          ReactionHolder(std::vector<stochCoeff>{stochCoeff(reag, 1)}, prod),
          Vm, Km) {}

T ReactionMichaelisMenten::BaseRate(T dt) { return Vm * dt; }

__device__ void ReactionMichaelisMenten::Rate(const T &reagent, T &progress) {
    progress *= reagent / (Km + reagent);
}

void ReactionMichaelisMenten::Print() {
    Reaction::Print();
    std::cout << "Vm = " << Vm << " ; Km = " << Km << "\n";
}