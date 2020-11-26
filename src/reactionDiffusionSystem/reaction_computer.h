#include <cuda_runtime.h>

#include "constants.hpp"
#include "dataStructures/array.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "helper/cuda/cuda_thread_manager.hpp"
#include "reaction_computer.hpp"
#include "system.hpp"

std::pair<int *, int> getRawCoeffs(State &state,
                                   std::vector<stochCoeff> &reactionSide) {
    int n = 0;
    for (auto reagent : reactionSide)
        n += reagent.second;
    int i = 0;
    int *raw_coeffs = new int[n];
    for (auto reagent : reactionSide)
        for (int k = 0; k < reagent.second; k++) {
            raw_coeffs[i] = state.names.at(reagent.first);
            i++;
        }
    int *d_raw_coeffs;
    gpuErrchk(cudaMalloc(&d_raw_coeffs, n * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_raw_coeffs, raw_coeffs, n * sizeof(int),
                         cudaMemcpyHostToDevice));
    delete raw_coeffs;
    return std::pair<int *, int>(d_raw_coeffs, n);
}

template <typename ReactionType>
__global__ void ComputeReactionK(D_Vector **state, int n_species, int *reagents,
                                 int n_reagents, int *products, int n_products,
                                 T base_rate, ReactionType &rate) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= state[0]->n)
        return;
    T progress = base_rate;
    for (int k = 0; k < n_reagents; k++)
        rate.Rate(state[reagents[k]]->At(i), progress);
    for (int k = 0; k < n_reagents; k++)
        state[reagents[k]]->At(i) -= progress;
    for (int k = 0; k < n_products; k++)
        state[products[k]]->At(i) += progress;
    return;
}

template <typename ReactionType>
void ComputeReaction(State &state, ReactionType &reaction, T dt) {
    auto tb = Make1DThreadBlock(state.size);
    auto d_state = state.GetDeviceState();
    auto products = getRawCoeffs(state, reaction.Products);
    auto reagents = getRawCoeffs(state, reaction.Reagents);
    ComputeReactionK<<<tb.block, tb.thread>>>(
        d_state, state.data.size(), reagents.first, reagents.second,
        products.first, products.second, reaction.BaseRate(dt),
        *reaction._device);
    gpuErrchk(cudaDeviceSynchronize());
    cudaFree(reagents.first);
    // state.FreeDeviceState(d_state);
}