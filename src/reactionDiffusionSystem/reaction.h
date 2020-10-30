#pragma once

// #include <nvfunctional>
#include <string>
#include <vector>

#include "constants.hpp"
#include "dataStructures/hd_data.hpp"
#include "dataStructures/sparse_matrix.hpp"
#include "matrixOperations/basic_operations.hpp"

typedef std::pair<std::string, int> stochCoeff;
typedef std::tuple<std::vector<stochCoeff>, std::vector<stochCoeff>, T>
    Reaction;

// Michaelis-Menten Reactions
typedef std::tuple<std::string, std::vector<stochCoeff>, T, T> MMReaction;