#include "reaction.hpp"

__device__ void PrintBody(const Reaction &reac) {
    printf("Reaction Data:\n\t");
    reac.Reagents.Print();
    printf("\t");
    reac.ReagentsCoeff.Print();
    printf("\t");
    reac.Products.Print();
    printf("\t");
    reac.ProductsCoeff.Print();
}

///////////
/// Debug puropse
//
__global__ void PrintK(ReactionMassAction &reac) { PrintBody(reac); }
__global__ void PrintK(ReactionMichaelisMenten &reac) { PrintBody(reac); }