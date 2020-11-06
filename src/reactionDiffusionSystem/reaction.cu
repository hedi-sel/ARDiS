#include "reaction.hpp"

Reaction::Reaction(std::vector<stochCoeff> reag, std::vector<stochCoeff> prod)
    : Reagents(reag), Products(prod) {}

void Reaction::Print() {
    for (auto coeff : Reagents)
        std::cout << coeff.second << "." << coeff.first << " + ";
    std::cout << "-> ";
    for (auto coeff : Products)
        std::cout << coeff.second << "." << coeff.first << " + ";
    std::cout << "\n";
}

/////// Mass Action

ReactionMassAction::ReactionMassAction(std::vector<stochCoeff> reag,
                                       std::vector<stochCoeff> prod, T rate)
    : Reaction(reag, prod), K(rate) {
    gpuErrchk(cudaMalloc(&_device, sizeof(ReactionMassAction)));
    gpuErrchk(cudaMemcpy(_device, this, sizeof(ReactionMassAction),
                         cudaMemcpyHostToDevice));
}

ReactionMassAction::ReactionMassAction(Reaction reac, T rate)
    : ReactionMassAction(reac.Reagents, reac.Products, rate) {}

T ReactionMassAction::BaseRate(T dt) { return K * dt; }

__device__ void ReactionMassAction::Rate(const T &reagent, T &progress) {
    progress *= reagent;
}

void ReactionMassAction::Print() {
    Reaction::Print();
    std::cout << "k=" << K << "\n";
}

/////// Michaleis Menten

ReactionMichaelisMenten::ReactionMichaelisMenten(Reaction reac, T Vm, T Km)
    : ReactionMichaelisMenten(reac.Reagents.at(0).first, reac.Products, Vm,
                              Km) {}

ReactionMichaelisMenten::ReactionMichaelisMenten(std::string reag,
                                                 std::vector<stochCoeff> prod,
                                                 T Vm, T Km)
    : Reaction(std::vector<stochCoeff>{stochCoeff(reag, 1)}, prod), Vm(Vm),
      Km(Km) {
    // gpuErrchk(cudaMalloc(&d_Km, sizeof(T)));
    // gpuErrchk(cudaMemcpy(d_Km, &Km, sizeof(T), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&_device, sizeof(ReactionMichaelisMenten)));
    gpuErrchk(cudaMemcpy(_device, this, sizeof(ReactionMichaelisMenten),
                         cudaMemcpyHostToDevice));
}

T ReactionMichaelisMenten::BaseRate(T dt) { return Vm * dt; }

__device__ void ReactionMichaelisMenten::Rate(const T &reagent, T &progress) {
    progress *= reagent / (Km + reagent);
}

void ReactionMichaelisMenten::Print() {
    Reaction::Print();
    std::cout << "Vm = " << Vm << " ; Km = " << Km << "\n";
}