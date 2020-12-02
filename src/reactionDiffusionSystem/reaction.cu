#include <cuda_runtime.h>

#include "reaction.hpp"

#include "reactionHelper.h"
reaction_holder::reaction_holder(std::vector<stochCoeff> reag,
                                 std::vector<stochCoeff> prod)
    : Reagents(reag), Products(prod) {}

reaction::reaction(std::map<std::string, int> &names, reaction_holder holder,
                   long unsigned size)
    : Holder(holder), Reagents(holder.Reagents.size()),
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

reaction::reaction(std::map<std::string, int> &names,
                   std::vector<stochCoeff> reag, std::vector<stochCoeff> prod,
                   long unsigned size)
    : reaction(names, reaction_holder(reag, prod), size) {}

__device__ __host__ void reaction::print() const {
#ifndef __CUDA_ARCH__
    for (auto coeff : Holder.Reagents)
        std::cout << coeff.second << "." << coeff.first << " + ";
    std::cout << "-> ";
    for (auto coeff : Holder.Products)
        std::cout << coeff.second << "." << coeff.first << " + ";
    std::cout << "\n";
#else
    printf("Warning: print has been called in base class reaction \n");
#endif
}

reaction::reaction(reaction &&other)
    : Holder(other.Holder), Reagents(std::move(other.Reagents)),
      ReagentsCoeff(std::move(other.ReagentsCoeff)),
      Products(std::move(other.Products)),
      ProductsCoeff(std::move(other.ProductsCoeff)) {}

/////// Mass Action

reaction_mass_action::reaction_mass_action(std::map<std::string, int> &names,
                                           std::vector<stochCoeff> reag,
                                           std::vector<stochCoeff> prod, T rate)
    : reaction_mass_action(names, reaction_holder(reag, prod), rate) {}

reaction_mass_action::reaction_mass_action(std::map<std::string, int> &names,
                                           reaction_holder reac, T rate)
    : reaction(names, reac, sizeof(reaction_mass_action)), K(rate) {
    gpuErrchk(cudaMalloc(&_device, sizeof(reaction_mass_action)));
    gpuErrchk(cudaMemcpy(_device, this, sizeof(reaction_mass_action),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
}

__host__ __device__ void reaction_mass_action::print() const {
#ifndef __CUDA_ARCH__
    reaction::print();
    std::cout << "k=" << K << "\n";
#else
    PrintBody(*this);
#endif
}

/////// Michaleis Menten

reaction_michaelis_menten::reaction_michaelis_menten(
    std::map<std::string, int> &names, reaction_holder reac, T Vm, T Km)
    : reaction(names, reac, sizeof(reaction_michaelis_menten)), Vm(Vm), Km(Km) {
    gpuErrchk(cudaMalloc(&_device, sizeof(reaction_michaelis_menten)));
    gpuErrchk(cudaMemcpy(_device, this, sizeof(reaction_michaelis_menten),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
}

reaction_michaelis_menten::reaction_michaelis_menten(
    std::map<std::string, int> &names, std::string reag,
    std::vector<stochCoeff> prod, T Vm, T Km)
    : reaction_michaelis_menten(
          names,
          reaction_holder(std::vector<stochCoeff>{stochCoeff(reag, 1)}, prod),
          Vm, Km) {}

__host__ __device__ void reaction_michaelis_menten::print() const {
#ifndef __CUDA_ARCH__
    reaction::print();
    std::cout << "Vm = " << Vm << " ; Km = " << Km << "\n";
#else
    PrintBody(*this);
#endif
}

reaction::~reaction() {}