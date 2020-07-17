#pragma once

#include <string>
#include <vector>

#include "constants.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "matrixOperations/basic_operations.hpp"
#include "solvers/conjugate_gradient_solver.hpp"
#include "state.hpp"

typedef std::pair<std::string, int> stochCoeff;
typedef std::tuple<std::vector<stochCoeff>, std::vector<stochCoeff>, T>
    Reaction;

// A top-level class that handles the operations for the reaction-diffusion
// simulation.

class System {
  public:
    // Data holder for the species spatial concentraions
    State state;

    CGSolver solver;
    D_Array b;

    // The set of reactions
    std::vector<Reaction> reactions;

    // Diffusion matrices
    D_SparseMatrix *damp_mat = nullptr;
    D_SparseMatrix *stiff_mat = nullptr;
    D_SparseMatrix diffusion_matrix;

    // Parameters
    T epsilon = 1e-3;
    T last_used_dt = 0;

    // Profiler
    ChronoProfiler profiler;

    // Give as input the size of the concentration vectors
    System(int);
    ~System();

    // Adds a new reaction
    void AddReaction(std::string reag, int kr, std::string prod, int kp,
                     T rate);
    void AddReaction(std::vector<stochCoeff> input,
                     std::vector<stochCoeff> output, T rate);
    void AddReaction(const std::string &reaction, T rate);
    void AddReaction(Reaction reaction);

    // Get the memory location of the dampness and stiffness matrices
    void LoadDampnessMatrix(D_SparseMatrix &damp_mat);
    void LoadStiffnessMatrix(D_SparseMatrix &stiff_mat);

    // Make one iteration of either rection or diffusion, for the given timestep
    // Note: For optimal speed, try do the diffusion iterations with the same
    //    time-step
    void IterateReaction(T dt, bool degradation = false);
    bool IterateDiffusion(T dt);
    void Prune(T value = 0);

    // Set the convergence threshold for the conjugae gradient method
    void SetEpsilon(T epsilon);

    void Print(int = 5);
};