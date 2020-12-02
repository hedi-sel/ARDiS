#pragma once

#include <string>
#include <vector>

#include "constants.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "matrixOperations/basic_operations.hpp"
#include "reaction.hpp"
#include "solvers/conjugate_gradient_solver.hpp"
#include "state.hpp"

// A top-level class that handles the operations for the reaction-diffusion
// simulation.

class simulation {
  public:
    // Data holder for the species spatial concentraions
    state current_state;

    cg_solver solver;
    d_vector b;

    // The set of reactions
    std::vector<reaction_mass_action> reactions;

    // The set of Michaelis-Menten Reactions
    std::vector<reaction_michaelis_menten> mmreactions;

    // Diffusion matrices
    d_spmatrix *damp_mat = nullptr;
    d_spmatrix *stiff_mat = nullptr;
    d_spmatrix diffusion_matrix;

    // Parameters
    T epsilon = 1e-3;
    T last_used_dt = 0;
    T drain = 1.e-13;

    // Enlapsed time
    T t = 0;

#ifndef NDEBUG_PROFILING
    // Profiler
    ChronoProfiler profiler;
#endif

    // Give as input the size of the concentration vectors
    simulation(int);
    ~simulation();

    // Adds a new reaction
    void add_reaction(std::string reag, int kr, std::string prod, int kp,
                      T rate);
    void add_reaction(const std::string &reaction, T rate);

    void add_mm_reaction(std::string reag, std::string prod, int kp, T Vm,
                         T Km);
    void add_mm_reaction(const std::string &reaction, T Vm, T Km);

    // Get the memory location of the dampness and stiffness matrices
    void load_dampness_matrix(d_spmatrix &damp_mat);
    void load_stiffness_matrix(d_spmatrix &stiff_mat);

    // Make one iteration of either rection or diffusion, for the given timestep
    // Note: For optimal speed, try do the diffusion iterations with the same
    //    time-step
    void iterate_reaction(T dt, bool degradation = false);
    bool iterate_diffusion(T dt);
    void prune(T value = 0);

    // Set the convergence threshold for the conjugae gradient method
    void SetEpsilon(T epsilon);
    void SetDrain(T drain);

    void print(int = 5);
};