#include <cstdio>
#include <fstream>

#include "conjugate_gradient_solver.hpp"
#include "constants.hpp"
#include "helper/chrono_profiler.hpp"

cg_solver::cg_solver(int n) : q(n), r(n), p(n) {}

bool cg_solver::cg_solve(d_spmatrix &d_mat, d_vector &b, d_vector &x, T epsilon,
                         std::string outputPath) {
#ifndef NDEBUG_PROFILING
    profiler.start("Preparing Data");
#endif
    dot(d_mat, x, q, true);

    r = d_vector(b); // Copy b into r
    alpha() = -1.0;
    alpha.update_dev();
    vector_sum(r, q, alpha(true), r);

    p = d_vector(r); // copy r into p
    beta() = 0.0;
    beta.update_dev();
    value() = 0.0;
    value.update_dev();
    dot(r, r, diff(true), true);
    diff.update_host();

    T diff0 = diff();

    int n_iter = 0;
    do {
        n_iter++;
#ifndef NDEBUG_PROFILING
        profiler.start("MatMult");
#endif
        dot(d_mat, p, q, true);
#ifndef NDEBUG_PROFILING
        profiler.start("VectorDot");
#endif
        dot(q, p, value(true), true);

        value.update_host();
        if (value() != 0)
            alpha() = diff() / value();
        else {
            n_iter_last = n_iter;
            return true;
        }
        alpha.update_dev();
#ifndef NDEBUG_PROFILING
        profiler.start("vector_sum");
#endif
        vector_sum(x, p, alpha(true), x, true);

        value() = -alpha();
        value.update_dev();

#ifndef NDEBUG_PROFILING
        profiler.start("vector_sum");
#endif
        vector_sum(r, q, value(true), r, true);

        value.set(&diff());

#ifndef NDEBUG_PROFILING
        profiler.start("VectorDot");
#endif
        dot(r, r, diff(true), true);

        diff.update_host();
        if (diff() != 0)
            beta() = diff() / value();
        else {
            n_iter_last = n_iter;
            return true;
        }
        beta.update_dev();

#ifndef NDEBUG_PROFILING
        profiler.start("vector_sum");
#endif
        vector_sum(r, p, beta(true), p, true);
    } while (diff() > epsilon * epsilon * diff0 && n_iter < 1000);
#ifndef NDEBUG_PROFILING
    profiler.end();
#endif

    n_iter_last = n_iter;

    if (diff() > epsilon * epsilon * diff0)
        printf("Warning: It did not converge\n");

    return !(diff() > epsilon * epsilon * diff0);
}

bool cg_solver::st_cg_solve(d_spmatrix &d_mat, d_vector &b, d_vector &x,
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
    diff.update_host();

    T diff0 = diff();

    int n_iter = 0;
    do {
        n_iter++;
        dot(d_mat, p, q, true);

        dot(q, p, value(true), true);
        value.update_host();
        if (value() != 0)
            alpha() = diff() / value();
        else
            alpha() = 0;
        alpha.update_dev();

        vector_sum(x, p, alpha(true), x, true);

        value() = -alpha();
        value.update_dev();
        vector_sum(r, q, value(true), r, true);

        value.set(&diff());
        dot(r, r, diff(true), true);
        diff.update_host();
        if (value() != 0)
            beta() = diff() / value();
        else
            beta() = 0;
        beta.update_dev();

        vector_sum(r, p, beta(true), p, true);
    } while (diff() > epsilon * epsilon * diff0 && n_iter < 1000);

    if (diff() > epsilon * epsilon * diff0)
        printf("Warning: It did not converge\n");

    return !(diff() > epsilon * epsilon * diff0);
}
