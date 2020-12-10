#include <cuda_runtime.h>

#include "constants.hpp"
#include "dataStructures/array.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "helper/cuda/cuda_thread_manager.hpp"
#include "simulation.hpp"

template <typename ReactionType>
__global__ void compute_reactionK(d_array<d_vector *> &state, T dt,
                                  ReactionType &rate) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= state.at(0)->n)
        return;
    rate.ApplyReaction(state, i, dt);
}
