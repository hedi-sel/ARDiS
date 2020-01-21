#include "conjugate_gradient_solver.hpp"
#include "constants.hpp"

D_SparseMatrix identity(0, 0);

void CGSolve(D_SparseMatrix &d_mat, D_Array &b, D_Array &x, T epsilon) {
    CGSolve(d_mat, b, x, epsilon, identity);
}

void CGSolve(D_SparseMatrix &d_mat, D_Array &b, D_Array &x, T epsilon,
             D_SparseMatrix &precond) {
    D_Array r(b);
    D_Array p(b);
    D_Array q(b.n, true);
    HDData<T> value;

    HDData<T> alpha(0.0);
    HDData<T> alphaDupl(0.0);
    HDData<T> beta(0.0);

    HDData<T> diff(0.0);
    Dot(r, p, diff(true), true);
    diff.SetHost();

    int n_iter = 0;
    do {
        n_iter++;
        Dot(d_mat, p, q, true);

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

        printf("Prelim result %i : %f\n", n_iter, diff());
    } while (diff() > epsilon * epsilon && n_iter < 1000);
    printf("N Iterations = %i : %f\n", n_iter, diff());
}