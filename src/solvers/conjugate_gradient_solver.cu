#include <cstdio>
#include <fstream>

#include "conjugate_gradient_solver.hpp"
#include "constants.hpp"
#include "helper/chrono_profiler.hpp"

cg_solver::cg_solver(int n) : q(n), r(n), p(n) {}

void ToCSV(d_vector &array, std::string outputPath, std::string label = "") {
    std::ofstream fout;
    fout.open(outputPath, std::ios_base::app);
    d_vector h_Arr(array, true);
    fout << label;
    for (size_t j = 0; j < h_Arr.n; j++)
        fout << ((j == 0) ? "" : "\t") << h_Arr.data[j];
    fout << "\n";
    fout.close();
}

void ToCSV(hd_data<T> &val, std::string outputPath, std::string label = "") {
    std::ofstream fout;
    fout.open(outputPath, std::ios_base::app);
    fout << label;
    fout << val();
    fout << "\n";
    fout.close();
}

bool cg_solver::CGSolve(d_spmatrix &d_mat, d_vector &b, d_vector &x, T epsilon,
                        std::string outputPath) {
#ifndef NDEBUG_PROFILING
    profiler.Start("Preparing Data");
#endif
    dot(d_mat, x, q, true);

    r = d_vector(b); // Copy b into r
    alpha() = -1.0;
    alpha.SetDevice();
    vector_sum(r, q, alpha(true), r);

    p = d_vector(r); // copy r into p
    beta() = 0.0;
    beta.SetDevice();
    value() = 0.0;
    value.SetDevice();
    dot(r, r, diff(true), true);
    diff.SetHost();

    T diff0 = diff();

    int n_iter = 0;
    do {
        n_iter++;
#ifndef NDEBUG_PROFILING
        profiler.Start("MatMult");
#endif
        dot(d_mat, p, q, true);
#ifndef NDEBUG_PROFILING
        profiler.Start("VectorDot");
#endif
        dot(q, p, value(true), true);

        value.SetHost();
        if (value() != 0)
            alpha() = diff() / value();
        else {
            n_iter_last = n_iter;
            return true;
        }
        alpha.SetDevice();
#ifndef NDEBUG_PROFILING
        profiler.Start("vector_sum");
#endif
        vector_sum(x, p, alpha(true), x, true);

        value() = -alpha();
        value.SetDevice();

#ifndef NDEBUG_PROFILING
        profiler.Start("vector_sum");
#endif
        vector_sum(r, q, value(true), r, true);

        value.Set(&diff());

#ifndef NDEBUG_PROFILING
        profiler.Start("VectorDot");
#endif
        dot(r, r, diff(true), true);

        diff.SetHost();
        if (diff() != 0)
            beta() = diff() / value();
        else {
            n_iter_last = n_iter;
            return true;
        }
        beta.SetDevice();

#ifndef NDEBUG_PROFILING
        profiler.Start("vector_sum");
#endif
        vector_sum(r, p, beta(true), p, true);
    } while (diff() > epsilon * epsilon * diff0 && n_iter < 1000);
#ifndef NDEBUG_PROFILING
    profiler.End();
#endif

    n_iter_last = n_iter;

    if (diff() > epsilon * epsilon * diff0)
        printf("Warning: It did not converge\n");

    return !(diff() > epsilon * epsilon * diff0);
}

bool cg_solver::StaticCGSolve(d_spmatrix &d_mat, d_vector &b, d_vector &x,
                              T epsilon) {
    d_vector q(b.n, true);
    dot(d_mat, x, q, true);

    d_vector r(b);
    hd_data<T> alpha(-1.0);
    vector_sum(r, q, alpha(true), r);

    d_vector p(r);
    hd_data<T> value;
    hd_data<T> beta(0.0);

    hd_data<T> diff(0.0);
    dot(r, r, diff(true), true);
    diff.SetHost();

    T diff0 = diff();

    int n_iter = 0;
    do {
        n_iter++;
        dot(d_mat, p, q, true);

        dot(q, p, value(true), true);
        value.SetHost();
        if (value() != 0)
            alpha() = diff() / value();
        else
            alpha() = 0;
        alpha.SetDevice();

        vector_sum(x, p, alpha(true), x, true);

        value() = -alpha();
        value.SetDevice();
        vector_sum(r, q, value(true), r, true);

        value.Set(&diff());
        dot(r, r, diff(true), true);
        diff.SetHost();
        if (value() != 0)
            beta() = diff() / value();
        else
            beta() = 0;
        beta.SetDevice();

        vector_sum(r, p, beta(true), p, true);
    } while (diff() > epsilon * epsilon * diff0 && n_iter < 1000);

    if (diff() > epsilon * epsilon * diff0)
        printf("Warning: It did not converge\n");

    return !(diff() > epsilon * epsilon * diff0);
}
