#include "reaction_computer.h"
#include "system.hpp"

System::System(int size) : state(size){};

void System::AddReaction(std::vector<stochCoeff> input,
                         std::vector<stochCoeff> output, T factor) {
    AddReaction(Reaction(input, output, factor));
}
void System::AddReaction(Reaction reaction) { reactions.push_back(reaction); };

void System::LoadDampnessMatrix(D_SparseMatrix &damp_mat) {
    this->damp_mat = &damp_mat;
}
void System::LoadStiffnessMatrix(D_SparseMatrix &stiff_mat) {
    this->stiff_mat = &stiff_mat;
}

__global__ void PruneK(D_Array **state, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= state[0]->n)
        return;
    for (int k = 0; k < size; k++) {
        if (state[k]->vals[i] < 0)
            state[k]->vals[i] = 0;
    }
}

void System::Prune(T value) {
    auto tb = Make1DThreadBlock(state.size);
    for (auto &vect : state.data)
        vect.Prune(value);
}

void System::IterateReaction(T dt) {
    for (auto reaction : reactions) {
        ConsumeReaction(state, reaction, std::get<2>(reaction) * dt);
    }
}

void System::IterateDiffusion(T dt) {
    if (damp_mat == nullptr || stiff_mat == nullptr) {
        printf("Error! Stiffness and Dampness matrices not loaded\n");
        return;
    }
    if (last_used_dt != dt) {
        printf("Building a diffusion matrix for dt = %f ... ", dt);
        HDData<T> m(-dt);
        MatrixSum(*damp_mat, *stiff_mat, m(true), diffusion_matrix);
        printf("Done!\n");
        last_used_dt = dt;
    }
    for (auto &species : state.data) {
        D_Array d_b(state.size);
        Dot(*damp_mat, species, d_b);
        CGSolve(diffusion_matrix, d_b, species, epsilon);
    }
}

void System::Print(int printCount) {
    state.Print(printCount);
    for (int i = 0; i < reactions.size(); i++) {
        for (auto coeff : std::get<0>(reactions.at(i)))
            std::cout << coeff.second << "." << coeff.first << " + ";
        std::cout << "-> ";
        for (auto coeff : std::get<1>(reactions.at(i)))
            std::cout << coeff.second << "." << coeff.first << " + ";
        std::cout << "k=" << std::get<2>(reactions.at(i)) << "\n";
    }
}
