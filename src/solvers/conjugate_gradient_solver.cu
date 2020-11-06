#include <cstdio>
#include <fstream>

#include "conjugate_gradient_solver.hpp"
#include "constants.hpp"
#include "hediHelper/chrono_profiler.hpp"

CGSolver::CGSolver(int n) : q(n), r(n), p(n) {}

void ToCSV(D_Vector &array, std::string outputPath, std::string label = "") {
    std::ofstream fout;
    fout.open(outputPath, std::ios_base::app);
    // fout << stateName << "\t" << state.data.size() << "\n";
    D_Vector h_Arr(array, true);
    // fout << species.first ;
    fout << label;
    for (size_t j = 0; j < h_Arr.n; j++)
        fout << ((j == 0) ? "" : "\t") << h_Arr.data[j];
    fout << "\n";
    fout.close();
}

void ToCSV(HDData<T> &val, std::string outputPath, std::string label = "") {
    std::ofstream fout;
    fout.open(outputPath, std::ios_base::app);
    // fout << stateName << "\t" << state.data.size() << "\n";
    // fout << species.first ;
    fout << label;
    fout << val();
    fout << "\n";
    fout.close();
}

bool CGSolver::CGSolve(D_SparseMatrix &d_mat, D_Vector &b, D_Vector &x,
                       T epsilon, std::string outputPath) {
#ifndef NDEBUG_PROFILING
    profiler.Start("Preparing Data");
#endif
    Dot(d_mat, x, q, true);

    r = b; // Copy b into r
    alpha() = -1.0;
    alpha.SetDevice();
    VectorSum(r, q, alpha(true), r);

    p = r; // copy r into p
    beta() = 0.0;
    beta.SetDevice();
    value() = 0.0;
    value.SetDevice();
    Dot(r, r, diff(true), true);
    diff.SetHost();

    // printf("values %f %f %f\n", alpha(), beta(), diff());

    T diff0 = diff();

    int n_iter = 0;
    do {
        n_iter++;
#ifndef NDEBUG_PROFILING
        profiler.Start("MatMult");
#endif
        Dot(d_mat, p, q, true);
#ifndef NDEBUG_PROFILING
        profiler.Start("VectorDot");
#endif
        Dot(q, p, value(true), true);

        value.SetHost();
        if (value() != 0)
            alpha() = diff() / value();
        else {
            n_iter_last = n_iter;
            return true;
        }
        alpha.SetDevice();
#ifndef NDEBUG_PROFILING
        profiler.Start("VectorSum");
#endif
        VectorSum(x, p, alpha(true), x, true);

        value() = -alpha();
        value.SetDevice();

#ifndef NDEBUG_PROFILING
        profiler.Start("VectorSum");
#endif
        VectorSum(r, q, value(true), r, true);

        value.Set(&diff());

#ifndef NDEBUG_PROFILING
        profiler.Start("VectorDot");
#endif
        Dot(r, r, diff(true), true);

        diff.SetHost();
        if (diff() != 0)
            beta() = diff() / value();
        else {
            n_iter_last = n_iter;
            return true;
        }
        beta.SetDevice();

#ifndef NDEBUG_PROFILING
        profiler.Start("VectorSum");
#endif
        VectorSum(r, p, beta(true), p, true);
    } while (diff() > epsilon * epsilon * diff0 && n_iter < 1000);
#ifndef NDEBUG_PROFILING
    profiler.End();
#endif

    n_iter_last = n_iter;

    if (diff() > epsilon * epsilon * diff0)
        printf("Warning: It did not converge\n");

    return !(diff() > epsilon * epsilon * diff0);
}

bool CGSolver::StaticCGSolve(D_SparseMatrix &d_mat, D_Vector &b, D_Vector &x,
                             T epsilon) {
    D_Vector q(b.n, true);
    Dot(d_mat, x, q, true);

    D_Vector r(b);
    HDData<T> alpha(-1.0);
    VectorSum(r, q, alpha(true), r);

    D_Vector p(r);
    HDData<T> value;
    HDData<T> beta(0.0);

    HDData<T> diff(0.0);
    Dot(r, r, diff(true), true);
    diff.SetHost();

    T diff0 = diff();

    int n_iter = 0;
    do {
        n_iter++;
        Dot(d_mat, p, q, true);

        Dot(q, p, value(true), true);
        value.SetHost();
        if (value() != 0)
            alpha() = diff() / value();
        else
            alpha() = 0;
        alpha.SetDevice();

        VectorSum(x, p, alpha(true), x, true);

        value() = -alpha();
        value.SetDevice();
        VectorSum(r, q, value(true), r, true);

        value.Set(&diff());
        Dot(r, r, diff(true), true);
        diff.SetHost();
        if (value() != 0)
            beta() = diff() / value();
        else
            beta() = 0;
        beta.SetDevice();

        VectorSum(r, p, beta(true), p, true);
    } while (diff() > epsilon * epsilon * diff0 && n_iter < 1000);

    if (diff() > epsilon * epsilon * diff0)
        printf("Warning: It did not converge\n");

    return !(diff() > epsilon * epsilon * diff0);
}
