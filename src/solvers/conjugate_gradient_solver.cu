#include "conjugate_gradient_solver.hpp"
#include "constants.hpp"

void CGSolve(MatrixSparse &d_mat, VectorDense &b, VectorDense &x, T epsilon) {
    VectorDense r(b.n, true);
    cudaMemcpy(r.vals, b.vals, sizeof(T) * b.n, cudaMemcpyDeviceToDevice);
    VectorDense p(b.n, true);
    cudaMemcpy(p.vals, b.vals, sizeof(T) * b.n, cudaMemcpyDeviceToDevice);
    VectorDense q(b.n, true);
    HDData<T> value;

    HDData<T> alpha(0.0);
    HDData<T> alphaDupl(0.0);
    HDData<T> beta(0.0);

    HDData<T> diff(0.0);
    Dot(r, p, diff(true), false);
    diff.SetHost();
    do {
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
    } while (diff() > epsilon);
}