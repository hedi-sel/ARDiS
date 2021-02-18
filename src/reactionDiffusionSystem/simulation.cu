#include "dataStructures/helper/apply_operation.h"
#include "parse_reaction.hpp"
#include "reaction_computer.h"
#include "simulation.hpp"
#include <fstream>

simulation::simulation(int size) : current_state(size), solver(size), b(size){};
simulation::simulation(state &imp_state) : simulation(std::move(imp_state)){};
simulation::simulation(state &&imp_state)
    : current_state(std::move(imp_state)), solver(imp_state.vector_size),
      b(imp_state.vector_size){};

void check_reaction(simulation &sys, reaction_holder &reaction) {
    for (auto species : reaction.Reagents) {
        if (sys.current_state.names.find(species.first) ==
            sys.current_state.names.end()) {
            std::cout << "\"" << species.first << "\""
                      << "\n";
            throw std::invalid_argument("^ This species is invalid\n");
        }
    }
    for (auto species : reaction.Products) {
        if (sys.current_state.names.find(species.first) ==
            sys.current_state.names.end()) {
            std::cout << "\"" << species.first << "\""
                      << "\n";
            throw std::invalid_argument("^ This species is invalid\n");
        }
    }
}

void check_reaction(simulation &sys, std::vector<std::string> &names) {
    for (auto species : names) {
        if (sys.current_state.names.find(species) ==
            sys.current_state.names.end()) {
            std::cout << "\"" << species << "\""
                      << "\n";
            throw std::invalid_argument("^ This species is invalid\n");
        }
    }
}

void simulation::add_reaction(const std::string &descriptor, T rate) {
    auto holder = parse_reaction(descriptor);
    check_reaction(*this, holder);
    reactions.emplace_back(current_state.names, holder, rate);
}
void simulation::add_reaction(std::string reag, int kr, std::string prod,
                              int kp, T rate) {
    auto names = std::vector<std::string>();
    names.push_back(reag);
    names.push_back(prod);
    check_reaction(*this, names);

    std::vector<stochCoeff> input;
    std::vector<stochCoeff> output;
    input.push_back(std::pair<std::string, int>(reag, kr));
    output.push_back(std::pair<std::string, int>(prod, kp));
    reactions.emplace_back(current_state.names, input, output, rate);
}

void simulation::add_mm_reaction(std::string reag, std::string prod, int kp,
                                 T Vm, T Km) {
    auto names = std::vector<std::string>();
    names.push_back(reag);
    names.push_back(prod);
    check_reaction(*this, names);

    std::vector<stochCoeff> output;
    output.push_back(std::pair<std::string, int>(prod, kp));

    mmreactions.emplace_back(current_state.names, reag, output, Vm, Km);
}
void simulation::add_mm_reaction(const std::string &descriptor, T Vm, T Km) {
    reaction_holder reaction = parse_reaction(descriptor);
    if (reaction.Reagents.size() != 1 || reaction.Reagents.at(0).second != 1) {
        throw std::invalid_argument(
            "A Michaelis-Menten reaction takes only one species as reagent\n");
    }
    check_reaction(*this, reaction);
    mmreactions.emplace_back(current_state.names, reaction.Reagents.at(0).first,
                             reaction.Products, Vm, Km);
}

void simulation::load_dampness_matrix(d_spmatrix &damp_mat) {
    this->damp_mat = &damp_mat;
}
void simulation::load_stiffness_matrix(d_spmatrix &stiff_mat) {
    this->stiff_mat = &stiff_mat;
}

__global__ void PruneK(d_vector **state, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= state[0]->n)
        return;
    for (int k = 0; k < size; k++) {
        if (state[k]->data[i] < 0)
            state[k]->data[i] = 0;
    }
}

void simulation::prune(T value) {
    for (auto &vect : current_state.vector_holder)
        vect.prune(value);
}

void simulation::prune_under(T value) {
    for (auto &vect : current_state.vector_holder)
        vect.prune_under(value);
}

void simulation::iterate_reaction(T dt) {
#ifndef NDEBUG_PROFILING
    profiler.start("Reaction");
#endif
    T drainXdt = drain * dt;
    auto drainLambda = [drainXdt] __device__(T & x) { x -= drainXdt; };
    for (auto &species : current_state.vector_holder) {
        apply_func(species, drainLambda);
        species.prune();
    }
    auto tb = make1DThreadBlock(current_state.size());
    for (auto &reaction : reactions) {
        compute_reactionK<<<tb.block, tb.thread>>>(
            *current_state.get_device_data()._device, dt, *reaction._device);
    }
    for (auto &reaction : this->mmreactions) {
        compute_reactionK<<<tb.block, tb.thread>>>(
            *current_state.get_device_data()._device, dt, *reaction._device);
    }
#ifndef NDEBUG_PROFILING
    profiler.end();
#endif
}

bool simulation::iterate_diffusion(T dt) {
#ifndef NDEBUG_PROFILING
    profiler.start("Diffusion Initialization");
#endif
    if (damp_mat == nullptr || stiff_mat == nullptr) {
        printf("Error! Stiffness and Dampness matrices not loaded\n");
        return false;
    }
    if (last_used_dt != dt) {
        if (last_used_dt != 0)
            printf(
                "Warning! dt should be kept constant when iterating diffusion");
        hd_data<T> m(-dt);
        matrix_sum(*damp_mat, *stiff_mat, m(true), diffusion_matrix);
        last_used_dt = dt;
    }
    for (int i = 0; i < current_state.n_species(); i++) {
        auto &species = current_state.vector_holder.at(i);
        auto &option = current_state.options_holder.at(i);
        if (!option.diffusion)
            continue;

#ifndef NDEBUG_PROFILING
        profiler.start("Diffusion Initialization");
#endif
        dot(*damp_mat, species, b);
#ifndef NDEBUG_PROFILING
        profiler.start("Diffusion");
#endif
        if (!solver.cg_solve(diffusion_matrix, b, species, epsilon)) {
            printf("Warning: It did not converge at time %f\n", t);
            species.print(20);
            return false;
        }
    }
#ifndef NDEBUG_PROFILING
    profiler.end();
#endif

    t += dt;
    return true;
}

void simulation::print(int printCount) {
    current_state.print(printCount);
    for (auto &reaction : reactions) {
        reaction.print();
    }
    for (auto &reaction : this->mmreactions) {
        reaction.print();
    }
#ifndef NDEBUG_PROFILING
    std::cout << "Global Profiler : \n";
    profiler.print();
    std::cout << "Operation Profiler : \n";
    solver.profiler.print();
#endif
}

void simulation::SetEpsilon(T epsilon) { this->epsilon = epsilon; }
void simulation::SetDrain(T drain) { this->drain = drain; }

simulation::~simulation(){};
