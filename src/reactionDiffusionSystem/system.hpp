#pragma once

#include "constants.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "matrixOperations/basic_operations.hpp"
#include "solvers/conjugate_gradient_solver.hpp"
#include "state.hpp"

typedef std::pair<std::string, int> stochCoeff;
typedef std::pair<std::vector<stochCoeff>, std::vector<stochCoeff>> Reaction;

// A top-level class that handles the operations for the reaction-diffusion
// simulation.

class System {
  public:
    // Data holder for the species spatial concentraions
    State state;

    // The set of reactions
    std::vector<Reaction> reactions;
    std::vector<T> factors;

    // Diffusion matrices
    D_SparseMatrix *damp_mat = nullptr;
    D_SparseMatrix *stiff_mat = nullptr;
    D_SparseMatrix diffusion_matrix;

    // Parameters
    T epsilon = 1e-3;
    T last_used_dt = 0;

    System(int);

    void AddReaction(std::vector<stochCoeff> input,
                     std::vector<stochCoeff> output, T factor);
    void AddReaction(Reaction coeffs, T factor);

    void LoadDampnessMatrix(D_SparseMatrix &damp_mat);
    void LoadStiffnessMatrix(D_SparseMatrix &stiff_mat);

    void IterateReaction(T dt);
    void IterateDiffusion(T dt);

    void SetEpsilon();

    void Print(int = 5);
};