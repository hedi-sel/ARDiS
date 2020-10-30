#include "dataStructures/helper/apply_operation.h"
#include "parseReaction.hpp"
#include "reaction_computer.h"
#include "system.hpp"
#include <fstream>

System::System(int size) : state(size), solver(size), b(size){};

void System::AddReaction(const std::string &descriptor, T rate) {
    Reaction reaction = ParseReaction(descriptor);
    std::get<2>(reaction) = rate;
    AddReaction(reaction);
}
void System::AddReaction(std::string reag, int kr, std::string prod, int kp,
                         T rate) {
    std::vector<stochCoeff> input;
    std::vector<stochCoeff> output;
    input.push_back(std::pair<std::string, int>(reag, kr));
    output.push_back(std::pair<std::string, int>(prod, kp));
    AddReaction(Reaction(input, output, rate));
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

void System::AddMMReaction(std::string reag, std::string prod, int kp, T Vm,
                           T Km) {
    std::vector<stochCoeff> output;
    output.push_back(std::pair<std::string, int>(prod, kp));
    AddMMReaction(MMReaction(reag, output, Vm, Km));
}
void System::AddMMReaction(const std::string &descriptor, T Vm, T Km) {
    Reaction reaction = ParseReaction(descriptor);
    if (std::get<0>(reaction).size() != 1 ||
        std::get<0>(reaction).at(0).second != 1)
        throw std::invalid_argument(
            "A Michaelis-Menten reaction takes only one species as reagent\n");
    AddMMReaction(MMReaction(std::get<0>(reaction).at(0).first,
                             std::get<1>(reaction), Vm, Km));
}
void System::AddMMReaction(MMReaction reaction) {
    if (state.names.find(std::get<0>(reaction)) == state.names.end()) {
        std::cout << "\"" << std::get<0>(reaction) << "\""
                  << "\n";
        throw std::invalid_argument("^ This species is invalid\n");
    }
    for (auto species : std::get<1>(reaction)) {
        if (state.names.find(species.first) == state.names.end()) {
            std::cout << "\"" << species.first << "\""
                      << "\n";
            throw std::invalid_argument("^ This species is invalid\n");
        }
    }
    this->mmreactions.push_back(reaction);
}

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

void System::IterateReaction(T dt, bool degradation) {
#ifndef NDEBUG_PROFILING
    profiler.Start("Reaction");
#endif
    T drainXdt = drain * dt;
    auto drainLambda = [drainXdt] __device__(T & x) { x -= drainXdt; };
    for (auto species : state.data) {
        ApplyFunction(*species, drainLambda);
        species->Prune();
    }
    for (auto reaction : reactions) {
        ConsumeReaction(
            state, reaction, std::get<2>(reaction) * dt,
            [] __device__(const T &x, T &progress) { progress *= x; });
    }
    for (auto mmreac : this->mmreactions) {
        throw "Michaelis Menten reactiosn not implemented yet";
    }
#ifndef NDEBUG_PROFILING
    profiler.End();
#endif
}

bool System::IterateDiffusion(T dt) {
#ifndef NDEBUG_PROFILING
    profiler.Start("Diffusion Initialization");
#endif
    if (damp_mat == nullptr || stiff_mat == nullptr) {
        printf("Error! Stiffness and Dampness matrices not loaded\n");
        return false;
    }
    if (last_used_dt != dt) {
        printf("Building a diffusion matrix for dt = %f ... ", dt);
        HDData<T> m(-dt);
        MatrixSum(*damp_mat, *stiff_mat, m(true), diffusion_matrix);
        printf("Done!\n");
        last_used_dt = dt;
    }
    for (auto &species : state.data) {
#ifndef NDEBUG_PROFILING
        profiler.Start("Diffusion Initialization");
#endif
        Dot(*damp_mat, *species, b);
#ifndef NDEBUG_PROFILING
        profiler.Start("Diffusion");
#endif
        if (!solver.CGSolve(diffusion_matrix, b, *species, epsilon)) {
            printf("Warning: It did not converge at time %f\n", t);
            species->Print(20);
            return false;
        }
    }

#ifndef NDEBUG_PROFILING
    profiler.End();
#endif

    // std::ofstream fout;
    // fout.open("output/CgmIterCount", std::ios_base::app);
    // // std::cout << t << "\t" << solver.n_iter_last << "\n";
    // fout << t << "\t" << solver.n_iter_last << "\n";
    // fout.close();

    t += dt;
    return true;
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
#ifndef NDEBUG_PROFILING
    std::cout << "Global Profiler : \n";
    profiler.Print();
    std::cout << "Operation Profiler : \n";
    solver.profiler.Print();
#endif
}

void System::SetEpsilon(T epsilon) { this->epsilon = epsilon; }
void System::SetDrain(T drain) { this->drain = drain; }

System::~System(){
    // if (damp_mat != nullptr)
    //     delete damp_mat;
    // if (stiff_mat != nullptr)
    //     delete stiff_mat;
};
