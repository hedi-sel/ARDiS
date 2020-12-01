#include "dataStructures/helper/apply_operation.h"
#include "parseReaction.hpp"
#include "reaction_computer.h"
#include "system.hpp"
#include <fstream>

System::System(int size) : state(size), solver(size), b(size){};

void CheckReaction(System &sys, ReactionHolder &reaction) {
    for (auto species : reaction.Reagents) {
        if (sys.state.names.find(species.first) == sys.state.names.end()) {
            std::cout << "\"" << species.first << "\""
                      << "\n";
            throw std::invalid_argument("^ This species is invalid\n");
        }
    }
    for (auto species : reaction.Products) {
        if (sys.state.names.find(species.first) == sys.state.names.end()) {
            std::cout << "\"" << species.first << "\""
                      << "\n";
            throw std::invalid_argument("^ This species is invalid\n");
        }
    }
}

void CheckReaction(System &sys, std::vector<std::string> &names) {
    for (auto species : names) {
        if (sys.state.names.find(species) == sys.state.names.end()) {
            std::cout << "\"" << species << "\""
                      << "\n";
            throw std::invalid_argument("^ This species is invalid\n");
        }
    }
}

void System::AddReaction(const std::string &descriptor, T rate) {
    auto holder = ParseReaction(descriptor);
    CheckReaction(*this, holder);
    reactions.emplace_back(state.names, holder, rate);
}
void System::AddReaction(std::string reag, int kr, std::string prod, int kp,
                         T rate) {
    auto names = std::vector<std::string>();
    names.push_back(reag);
    names.push_back(prod);
    CheckReaction(*this, names);

    std::vector<stochCoeff> input;
    std::vector<stochCoeff> output;
    input.push_back(std::pair<std::string, int>(reag, kr));
    output.push_back(std::pair<std::string, int>(prod, kp));
    reactions.emplace_back(state.names, input, output, rate);
}

void System::AddMMReaction(std::string reag, std::string prod, int kp, T Vm,
                           T Km) {
    auto names = std::vector<std::string>();
    names.push_back(reag);
    names.push_back(prod);
    CheckReaction(*this, names);

    std::vector<stochCoeff> output;
    output.push_back(std::pair<std::string, int>(prod, kp));

    mmreactions.emplace_back(state.names, reag, output, Vm, Km);
}
void System::AddMMReaction(const std::string &descriptor, T Vm, T Km) {
    ReactionHolder reaction = ParseReaction(descriptor);
    if (reaction.Reagents.size() != 1 || reaction.Reagents.at(0).second != 1)
        throw std::invalid_argument(
            "A Michaelis-Menten reaction takes only one species as reagent\n");
    CheckReaction(*this, reaction);
    mmreactions.emplace_back(state.names, reaction.Reagents.at(0).first,
                             reaction.Products, Vm, Km);
}

void System::LoadDampnessMatrix(D_SparseMatrix &damp_mat) {
    this->damp_mat = &damp_mat;
}
void System::LoadStiffnessMatrix(D_SparseMatrix &stiff_mat) {
    this->stiff_mat = &stiff_mat;
}

__global__ void PruneK(D_Vector **state, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= state[0]->n)
        return;
    for (int k = 0; k < size; k++) {
        if (state[k]->data[i] < 0)
            state[k]->data[i] = 0;
    }
}

void System::Prune(T value) {
    for (auto &vect : state.vector_holder)
        vect.Prune(value);
}

void System::IterateReaction(T dt, bool degradation) {
#ifndef NDEBUG_PROFILING
    profiler.Start("Reaction");
#endif
    T drainXdt = drain * dt;
    auto drainLambda = [drainXdt] __device__(T & x) { x -= drainXdt; };
    for (auto &species : state.vector_holder) {
        ApplyFunction(species, drainLambda);
        species.Prune();
    }
    for (auto &reaction : reactions) {
        ComputeReaction<decltype(reaction)>(state, reaction, dt);
    }
    for (auto &reaction : this->mmreactions) {
        ComputeReaction<decltype(reaction)>(state, reaction, dt);
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
    for (auto &species : state.vector_holder) {
#ifndef NDEBUG_PROFILING
        profiler.Start("Diffusion Initialization");
#endif
        Dot(*damp_mat, species, b);
#ifndef NDEBUG_PROFILING
        profiler.Start("Diffusion");
#endif
        if (!solver.CGSolve(diffusion_matrix, b, species, epsilon)) {
            printf("Warning: It did not converge at time %f\n", t);
            species.Print(20);
            return false;
        }
    }
#ifndef NDEBUG_PROFILING
    profiler.End();
#endif

    t += dt;
    return true;
}

void System::Print(int printCount) {
    state.Print(printCount);
    for (auto &reaction : reactions) {
        reaction.Print();
    }
    for (auto &reaction : this->mmreactions) {
        reaction.Print();
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

System::~System(){};
