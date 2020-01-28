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
    D_Array p(b);
    D_Array r(b);
    D_Array q(b.n, true);
    Dot(d_mat, x, q, true);
    HDData<T> alpha(-1.0);
    VectorSum(r, q, alpha(true), r);
    HDData<T> value;
    HDData<T> alphaDupl(0.0);
    HDData<T> beta(0.0);

    HDData<T> diff(0.0);
    Dot(r, p, diff(true), true);
    diff.SetHost();

    int n_iter = 0;
    do {
        n_iter++;
        profiler.Start("MatMult");
        Dot(d_mat, p, q, true);
        profiler.End();

        profiler.Start("VectorDot");

        Dot(q, p, value(true), true);
        value.SetHost();
        alpha() = diff() / value();
        alpha.SetDevice();

        VectorSum(x, p, alpha(true), x, false);

        alphaDupl() = -alpha();
        alphaDupl.SetDevice();
        VectorSum(r, q, alphaDupl(true), r, true);

        Dot(r, q, diff(true), true);
        diff.SetHost();
        beta() = -diff() / value();
        beta.SetDevice();

        VectorSum(r, p, beta(true), p, true);

        Dot(r, p, diff(true), true);
        diff.SetHost();
        profiler.End();

        // printf("Prelim result %i : %f\n", n_iter, diff());
    } while (diff() > epsilon * epsilon && n_iter < 1000);
    profiler.Print();
    printf("\nN Iterations = %i : %f\n", n_iter, diff());
}