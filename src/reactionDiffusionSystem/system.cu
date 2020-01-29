#include "system.hpp"

System::System(int size) : state(size){};

void System::AddReaction(std::vector<stochCoeff> input,
                         std::vector<stochCoeff> output, T factor) {
    AddReaction(Reaction(input, output), factor);
}
void System::AddReaction(Reaction reaction, T factor) {
    reactions.push_back(reaction);
    factors.push_back(factor);
};

void System::LoadDampnessMatrix(D_SparseMatrix &damp_mat) {
    this->damp_mat = &damp_mat;
}
void System::LoadStiffnessMatrix(D_SparseMatrix &stiff_mat) {
    this->stiff_mat = &stiff_mat;
}

void System::IterateReaction(T dt) {}
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
        for (auto coeff : reactions.at(i).first)
            std::cout << coeff.first << "." << coeff.second << " + ";
        std::cout << "-> ";
        for (auto coeff : reactions.at(i).second)
            std::cout << coeff.first << "." << coeff.second << " + ";
        std::cout << "\n";
    }
}