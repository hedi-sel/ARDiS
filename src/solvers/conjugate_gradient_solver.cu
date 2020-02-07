#include "conjugate_gradient_solver.hpp"
#include "constants.hpp"
#include "hediHelper/chrono_profiler.hpp"

D_SparseMatrix identity(0, 0);

void CGSolve(D_SparseMatrix &d_mat, D_Array &b, D_Array &x, T epsilon) {
    CGSolve(d_mat, b, x, epsilon, identity);
}

void CGSolve(D_SparseMatrix &d_mat, D_Array &b, D_Array &x, T epsilon,
             D_SparseMatrix &precond) {
    ChronoProfiler profiler;
    profiler.Start("Preparing Data");
    D_Array q(b.n, true);
    Dot(d_mat, x, q, true);

    D_Array r(b);
    HDData<T> alpha(-1.0);
    VectorSum(r, q, alpha(true), r);

    D_Array p(r);
    HDData<T> value;
    HDData<T> beta(0.0);

    HDData<T> diff(0.0);
    Dot(r, r, diff(true), true);
    diff.SetHost();

    T diff0 = diff();

    int n_iter = 0;
    do {
        n_iter++;
        profiler.Start("MatMult");
        Dot(d_mat, p, q, true);
        profiler.End();

        profiler.Start("VectorDot");

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

        value.Set(diff());
        Dot(r, r, diff(true), true);
        diff.SetHost();
        if (value() != 0)
            beta() = diff() / value();
        else
            beta() = 0;
        beta.SetDevice();

        VectorSum(r, p, beta(true), p, true);

        profiler.End();
    } while (diff() > epsilon * epsilon * diff0 && n_iter < 1000);
    // profiler.Print();
    // printf("N Iterations = %i : %.3e\n", n_iter, diff());

    if (diff() > epsilon * epsilon * diff0)
        printf("Warning: It did not converge");
}