#include "reaction.hpp"

__device__ void PrintBody(const reaction &reac) {
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
__global__ void PrintK(reaction_mass_action &reac) { PrintBody(reac); }
__global__ void PrintK(reaction_michaelis_menten &reac) { PrintBody(reac); }