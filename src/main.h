#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <math.h>

#include "dataStructures/array.hpp"
#include "dataStructures/read_mtx_file.h"
#include "dataStructures/sparse_matrix.hpp"
#include "hediHelper/cuda/cusolverSP_error_check.h"
#include "hediHelper/cuda/cusparse_error_check.h"
#include "matrixOperations/basic_operations.hpp"
#include "matrixOperations/row_ordering.hpp"
#include "reactionDiffusionSystem/system.hpp"
#include "solvers/conjugate_gradient_solver.hpp"
#include "solvers/inversion_solver.h"

void ToCSV(State &state, std::string species, std::string outputPath) {
    std::ofstream fout;
    fout.open(outputPath, std::ios_base::app);
    // fout << stateName << "\t" << state.data.size() << "\n";
    D_Array h_Arr(*state.data.at(state.names.at(species)), true);
    // fout << species.first ;
    for (size_t j = 0; j < h_Arr.n; j++)
        fout << ((j == 0) ? "" : "\t") << h_Arr.vals[j];
    fout << "\n";
    fout.close();
}

bool LabyrinthExplore(std::string dampingPath, std::string stiffnessPath,
                      py::array_t<T> &u, T reaction, T max_time, T dt,
                      T plot_dt, std::string name, py::array_t<T> &mesh_x,
                      py::array_t<T> &mesh_y) {
    std::string outputPath = "./outputCsv/" + name + ".csv";
    remove(outputPath.c_str());

    bool isSuccess = true;
    D_SparseMatrix d_D(ReadFromFile(dampingPath), true);
    d_D.ConvertMatrixToCSR();
    D_SparseMatrix d_S(ReadFromFile(stiffnessPath), true);
    d_S.ConvertMatrixToCSR();
    ReadFromFile(dampingPath);
    ReadFromFile(stiffnessPath);
    System system(u.size());
    system.state.AddSpecies("N");
    system.state.AddSpecies("P");
    gpuErrchk(cudaMemcpy(system.state.GetSpecies("N").vals, u.data(),
                         sizeof(T) * u.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(system.state.GetSpecies("P").vals, u.data(),
                         sizeof(T) * u.size(), cudaMemcpyHostToDevice));
    D_Array MeshX(mesh_x.size());
    MeshX.name = "MeshX";
    D_Array MeshY(mesh_y.size());
    MeshY.name = "MeshY";
    gpuErrchk(cudaMemcpy(MeshX.vals, mesh_x.data(), sizeof(T) * mesh_x.size(),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(MeshY.vals, mesh_y.data(), sizeof(T) * mesh_y.size(),
                         cudaMemcpyHostToDevice));
    system.LoadDampnessMatrix(d_D);
    system.LoadStiffnessMatrix(d_S);
    system.AddReaction("N", 1, "N", 2, reaction);
    std::vector<stochCoeff> input;
    input.push_back(std::pair<std::string, int>("N", 1));
    input.push_back(std::pair<std::string, int>("P", 1));
    system.AddReaction(input, "P", 2, reaction);
    system.Print();

    int plotCount = 0;
    int progPrintCount = 0;

    int nTests = 5;
    T *angles = new T[nTests + 2];
    for (int i = 0; i < nTests + 2; i++)
        angles[i] = (M_PI / 2) * i /* (nTests + 1 - i) */ / (nTests + 1.0);
    TriangleZone *TestZones = new TriangleZone[nTests + 1];
    for (int i = 0; i < nTests + 1; i++)
        TestZones[i] =
            TriangleZone(0, 0, 10 * cos(angles[i]), 10 * sin(angles[i]),
                         10 * cos(angles[i + 1]), 10 * sin(angles[i + 1]));
    T *TestMinTime = new T[nTests];
    T *TestMaxTime = new T[nTests];
    int currentMinTest = 0;
    int currentMaxTest = 0;

    RectangleZone FinishZone = RectangleZone(0, 0, 0.2, 10);
    bool Finished = false;
    T FinishTime = 0;

    float threashold = 0.8;
    float timeAfterReaching = 3;

    for (int i = 0; i < max_time / dt; i++) {
        if (plotCount * plot_dt <= i * dt) {
            ToCSV(system.state, "N", outputPath);
            plotCount += 1;
        }
        if (!system.IterateDiffusion(dt, "./cry/" + name + ".csv"))
            isSuccess = false;
        system.IterateReaction(dt);
        // system.Print();

        // if (max_time / dt >= 100 && i >= progPrintCount * max_time / dt / 10
        // &&
        //     i < progPrintCount * max_time / dt / 10 + 1) {
        //     printf("%i completed \n", progPrintCount * 10);
        //     progPrintCount += 1;
        // }

        // while (currentMinTest < nTests &&
        //        GetMinZone(system.state.GetSpecies("N"), MeshX, MeshY,
        //                   TestZones[currentMinTest + 1]) > threashold) {
        //     printf("Min %i : %f\n", currentMinTest, i * dt);
        //     TestMinTime[currentMinTest] = i * dt;
        //     currentMinTest += 1;
        // }
        // while (currentMaxTest < nTests &&
        //        GetMaxZone(system.state.GetSpecies("N"), MeshX, MeshY,
        //                   TestZones[currentMaxTest]) > threashold) {
        //     printf("Max %i : %f\n", currentMaxTest, i * dt);
        //     TestMaxTime[currentMaxTest] = i * dt;
        //     currentMaxTest += 1;
        // }

        if (GetMinZone(system.state.GetSpecies("N"), MeshX, MeshY, FinishZone) >
            threashold) {
            FinishTime = i * dt;
            Finished = true;
        }

        if (Finished && i * dt >= FinishTime * timeAfterReaching)
            break;
    }

    if (Finished) {
        printf("Labyrinthe traversé! Arrivée à t= %f s \n", FinishTime);
        // print("Exterior times ", TestMinTime);
        // print("Interior times ", TestMaxTime);
    } else
        printf("Labyrinth not completed\n");

    delete[] angles;
    delete[] TestMaxTime;
    delete[] TestMinTime;
    delete[] TestZones;
    return isSuccess;
}

void checkSolve(D_SparseMatrix &M, D_Array &d_b, D_Array &d_x) {
    D_Array vec(d_b.n, true);
    Dot(M, d_x, vec, true);
    HDData<T> m1(-1);
    VectorSum(d_b, vec, m1(true), vec, true);
    Dot(vec, vec, m1(true), true);
    m1.SetHost();
    printf("Norme de la difference: %f\n", m1());
}

void SolveConjugateGradient(D_SparseMatrix &d_mat, D_Array &d_b, T epsilon,
                            D_Array &d_x) {
    CGSolver::StaticCGSolve(d_mat, d_b, d_x, epsilon);
#ifndef NDEBUG
    checkSolve(d_mat, d_x, d_b);
#endif
    // PrintDotProfiler();
}

void DiffusionTest(D_SparseMatrix &d_stiff, D_SparseMatrix &d_damp, T tau,
                   D_Array &d_u, T epsilon) {
    D_Array d_b(d_u.n, true);
    Dot(d_damp, d_u, d_b);
    D_SparseMatrix M(d_stiff.rows, d_stiff.cols, 0, COO, true);
    HDData<T> m(-tau);
    MatrixSum(d_damp, d_stiff, m(true), M);
    CGSolver::StaticCGSolve(M, d_b, d_u, epsilon);
}

D_Array SolveCholesky(D_SparseMatrix &d_mat, py::array_t<T> &bVec) {
    assert(bVec.size() == d_mat.rows);
    assert(d_mat.isDevice);

    D_Array b(bVec.size(), true);
    gpuErrchk(cudaMemcpy(b.vals, bVec.data(), sizeof(T) * b.n,
                         cudaMemcpyHostToDevice));
    D_Array x(d_mat.rows, true);
    solveLinEqBody(d_mat, b, x);

    return std::move(x);
}
