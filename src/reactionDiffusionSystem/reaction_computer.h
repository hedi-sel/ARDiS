#include <cuda_runtime.h>

#include "constants.hpp"
#include "dataStructures/array.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "hediHelper/cuda/cuda_thread_manager.hpp"
#include "reaction_computer.hpp"
#include "system.hpp"

std::pair<int *, int> getRawCoeffs(State &state,
                                   std::vector<stochCoeff> &reactionSide) {
    int n = 0;
    for (auto coef : reactionSide)
        n += coef.second;
    int i = 0;
    int *raw_coeffs = new int[n];
    for (auto coef : reactionSide)
        for (int k = 0; k < coef.second; k++) {
            raw_coeffs[i] = state.names.at(coef.first);
            i++;
        }
    int *d_raw_coeffs;
    gpuErrchk(cudaMalloc(&d_raw_coeffs, n * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_raw_coeffs, raw_coeffs, n * sizeof(int),
                         cudaMemcpyHostToDevice));
    delete raw_coeffs;
    return std::pair<int *, int>(d_raw_coeffs, n);
}

__global__ void ConsumeReactionK(D_Array **state, int n_species, int *reagents,
                                 int n_reagents, int *products, int n_products,
                                 T rate) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= state[0]->n)
        return;
    T progress = rate;
    for (int k = 0; k < n_reagents; k++)
        progress *= state[reagents[k]]->vals[i];
    for (int k = 0; k < n_reagents; k++)
        state[reagents[k]]->vals[i] -= progress;
    for (int k = 0; k < n_products; k++)
        state[products[k]]->vals[i] += progress;
    return;
}

void ConsumeReaction(State &state, Reaction &reaction, T rate) {
    auto tb = Make1DThreadBlock(state.size);
    auto d_state = state.GetDeviceState();
    auto reagents = getRawCoeffs(state, std::get<0>(reaction));
    auto products = getRawCoeffs(state, std::get<1>(reaction));
    ConsumeReactionK<<<tb.block, tb.thread>>>(
        d_state, state.data.size(), reagents.first, reagents.second,
        products.first, products.second, rate);
    gpuErrchk(cudaDeviceSynchronize());
    state.FreeDeviceState(d_state);
}