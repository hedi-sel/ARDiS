#include "reaction_computer.h"
#include "system.hpp"

System::System(int size) : state(size), solver(size), b(size){};

void System::AddReaction(std::string reag, int kr, std::string prod, int kp,
                         T rate) {
    std::vector<stochCoeff> input;
    std::vector<stochCoeff> output;
    input.push_back(std::pair<std::string, int>(reag, kr));
    output.push_back(std::pair<std::string, int>(prod, kp));
    AddReaction(input, output, rate);
}
void System::AddReaction(std::vector<stochCoeff> input, std::string prod,
                         int kp, T rate) {
    std::vector<stochCoeff> output;
    output.push_back(std::pair<std::string, int>(prod, kp));
    AddReaction(input, output, rate);
}
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
        vect->Prune(value);
}

void System::IterateReaction(T dt) {
    for (auto reaction : reactions) {
        ConsumeReaction(state, reaction, std::get<2>(reaction) * dt);
    }
}

int i = 0;
bool System::IterateDiffusion(T dt, std::string outPath) {
    bool isSuccess = true;
    if (damp_mat == nullptr || stiff_mat == nullptr) {
        printf("Error! Stiffness and Dampness matrices not loaded\n");
        return false;
    }
    if (last_used_dt != dt) {
        printf("Building a diffusion matrix for dt = %f ... ", dt);
        HDData<T> m(-dt);
        // if (diffusion_matrix != nullptr) {
        //     free(diffusion_matrix);
        // }
        // diffusion_matrix = new D_SparseMatrix();
        MatrixSum(*damp_mat, *stiff_mat, m(true), diffusion_matrix);
        // diffusion_matrix.Print(100);
        printf("Done!\n");
        last_used_dt = dt;

        if (outPath != std::string("")) {
            remove(outPath.c_str());
        }
    }
    for (auto &species : state.data) {
        b.name = "B";
        // D_Array copy(*species);
        Dot(*damp_mat, *species, b);
        // damp_mat->Print(20);
        // diffusion_matrix.Print(20);
        // d_b.Print(20);
        // printf("\n");
        // printf("\n");
        if (!solver.CGSolve(diffusion_matrix, b, *species, epsilon, outPath)) {
            printf("Warning: It did not converge on %i\n", i);
            // delete species;
            // species = new D_Array(copy);
            // Dot(*damp_mat, species, d_b);
            species->Print(20);
            isSuccess = false;
        }
    }
    i++;
    return isSuccess;
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

System::~System(){
    // if (damp_mat != nullptr)
    //     delete damp_mat;
    // if (stiff_mat != nullptr)
    //     delete stiff_mat;
};
