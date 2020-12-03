#include "reaction.hpp"

__device__ void printBody(const reaction &reac) {
    printf("Reaction Data:\n\t");
    reac.Reagents.print();
    printf("\t");
    reac.ReagentsCoeff.print();
    printf("\t");
    reac.Products.print();
    printf("\t");
    reac.ProductsCoeff.print();
}

///////////
/// Debug puropse
//
__global__ void printK(reaction_mass_action &reac) { printBody(reac); }
__global__ void printK(reaction_michaelis_menten &reac) { printBody(reac); }