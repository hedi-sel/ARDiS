#include "dataStructures/helper/apply_operation.h"
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
void System::AddReaction(std::vector<stochCoeff> input,
                         std::vector<stochCoeff> output, T factor) {
    AddReaction(Reaction(input, output, factor));
}
void System::AddReaction(Reaction reaction) {
    for (auto species : std::get<0>(reaction)) {
        if (state.names.find(species.first) == state.names.end()) {
            std::cout << "\"" << species.first << "\""
                      << "\n";
            throw std::invalid_argument("^ This species is invalid\n");
        }
    }
    for (auto species : std::get<1>(reaction)) {
        if (state.names.find(species.first) == state.names.end()) {
            std::cout << "\"" << species.first << "\""
                      << "\n";
            throw std::invalid_argument("^ This species is invalid\n");
        }
    }
    reactions.push_back(reaction);
};

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

const T drain = 1.e-15;

void System::IterateReaction(T dt, bool degradation) {
    profiler.Start("Reaction");
    auto drainLambda = [] __device__(T & x) { x -= drain; };
    for (auto species : state.data) {
        ApplyFunction(*species, drainLambda);
        species->Prune();
    }
    for (auto reaction : reactions) {
        ConsumeReaction(state, reaction, std::get<2>(reaction) * dt);
    }
    profiler.End();
}

int i = 0;
bool System::IterateDiffusion(T dt) {
    profiler.Start("Diffusion Initialization");
    if (damp_mat == nullptr || stiff_mat == nullptr) {
        printf("Error! Stiffness and Dampness matrices not loaded\n");
        return false;
    }
    profiler.End();
    if (last_used_dt != dt) {
        printf("Building a diffusion matrix for dt = %f ... ", dt);
        HDData<T> m(-dt);
        MatrixSum(*damp_mat, *stiff_mat, m(true), diffusion_matrix);
        printf("Done!\n");
        last_used_dt = dt;
    }
    profiler.Start("Diffusion");
    for (auto &species : state.data) {
        Dot(*damp_mat, *species, b);
        if (!solver.CGSolve(diffusion_matrix, b, *species, epsilon)) {
            printf("Warning: It did not converge on %i\n", i);
            species->Print(20);
            return false;
        }
    }
    i++;
    return true;
    profiler.End();
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
    std::cout << "Global Profiler : \n";
    profiler.Print();
    std::cout << "Operation Profiler : \n";
    solver.profiler.Print();
}

void System::SetEpsilon(T epsilon) { this->epsilon = epsilon; }

System::~System(){
    // if (damp_mat != nullptr)
    //     delete damp_mat;
    // if (stiff_mat != nullptr)
    //     delete stiff_mat;
};
