#include <cuda_runtime.h>

#include "reaction.hpp"

#include "reactionHelper.h"
ReactionHolder::ReactionHolder(std::vector<stochCoeff> reag,
                               std::vector<stochCoeff> prod)
    : Reagents(reag), Products(prod) {}

Reaction::Reaction(std::map<std::string, int> &names, ReactionHolder holder,
                   long unsigned size)
    : holder(holder), Reagents(holder.Reagents.size()),
      ReagentsCoeff(holder.Reagents.size()), Products(holder.Products.size()),
      ProductsCoeff(holder.Products.size()) {
    int reag_size = holder.Reagents.size();
    int reagents[reag_size];
    int reagents_coeff[reag_size];
    for (int i = 0; i < reag_size; i++) {
        reagents[i] = names.at(holder.Reagents.at(i).first);
        reagents_coeff[i] = holder.Reagents.at(i).second;
    }
    cudaMemcpy(Reagents.data, reagents, sizeof(int) * reag_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(ReagentsCoeff.data, reagents_coeff, sizeof(int) * reag_size,
               cudaMemcpyHostToDevice);

    int prod_size = holder.Products.size();
    int products[prod_size];
    int products_coeff[prod_size];
    for (int i = 0; i < prod_size; i++) {
        products[i] = names.at(holder.Products.at(i).first);
        products_coeff[i] = holder.Products.at(i).second;
    }
    cudaMemcpy(Products.data, products, sizeof(int) * prod_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(ProductsCoeff.data, products_coeff, sizeof(int) * prod_size,
               cudaMemcpyHostToDevice);

    // gpuErrchk(cudaMalloc(&_device, size));
    // gpuErrchk(cudaMemcpy(_device, this, size, cudaMemcpyHostToDevice));
}

Reaction::Reaction(std::map<std::string, int> &names,
                   std::vector<stochCoeff> reag, std::vector<stochCoeff> prod,
                   long unsigned size)
    : Reaction(names, ReactionHolder(reag, prod), size) {}

__device__ __host__ void Reaction::Print() const {
#ifndef __CUDA_ARCH__
    for (auto coeff : holder.Reagents)
        std::cout << coeff.second << "." << coeff.first << " + ";
    std::cout << "-> ";
    for (auto coeff : holder.Products)
        std::cout << coeff.second << "." << coeff.first << " + ";
    std::cout << "\n";
#else
    printf("Warning: Print has been called in base class Reaction \n");
#endif
}

Reaction::Reaction(Reaction &&other)
    : holder(other.holder), Reagents(std::move(other.Reagents)),
      ReagentsCoeff(std::move(other.ReagentsCoeff)),
      Products(std::move(other.Products)),
      ProductsCoeff(std::move(other.ProductsCoeff)) {}

/////// Mass Action

ReactionMassAction::ReactionMassAction(std::map<std::string, int> &names,
                                       std::vector<stochCoeff> reag,
                                       std::vector<stochCoeff> prod, T rate)
    : ReactionMassAction(names, ReactionHolder(reag, prod), rate) {}

ReactionMassAction::ReactionMassAction(std::map<std::string, int> &names,
                                       ReactionHolder reac, T rate)
    : Reaction(names, reac, sizeof(ReactionMassAction)), K(rate) {
    gpuErrchk(cudaMalloc(&_device, sizeof(ReactionMassAction)));
    gpuErrchk(cudaMemcpy(_device, this, sizeof(ReactionMassAction),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
}

__host__ __device__ void ReactionMassAction::Print() const {
#ifndef __CUDA_ARCH__
    Reaction::Print();
    std::cout << "k=" << K << "\n";
#else
    PrintBody(*this);
#endif
}

/////// Michaleis Menten

ReactionMichaelisMenten::ReactionMichaelisMenten(
    std::map<std::string, int> &names, ReactionHolder reac, T Vm, T Km)
    : Reaction(names, reac, sizeof(ReactionMichaelisMenten)), Vm(Vm), Km(Km) {
    gpuErrchk(cudaMalloc(&_device, sizeof(ReactionMichaelisMenten)));
    gpuErrchk(cudaMemcpy(_device, this, sizeof(ReactionMichaelisMenten),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
}

ReactionMichaelisMenten::ReactionMichaelisMenten(
    std::map<std::string, int> &names, std::string reag,
    std::vector<stochCoeff> prod, T Vm, T Km)
    : ReactionMichaelisMenten(
          names,
          ReactionHolder(std::vector<stochCoeff>{stochCoeff(reag, 1)}, prod),
          Vm, Km) {}

__host__ __device__ void ReactionMichaelisMenten::Print() const {
#ifndef __CUDA_ARCH__
    Reaction::Print();
    std::cout << "Vm = " << Vm << " ; Km = " << Km << "\n";
#else
    PrintBody(*this);
#endif
}

Reaction::~Reaction() {}