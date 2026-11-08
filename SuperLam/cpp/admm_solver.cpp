#include "admm_solver.h"
#include <iostream>
#include <cmath>

namespace lam {

void ADMMSolver::init(const CMatrix& A_, int cg_iters_, double rho_) {
    A = A_;
    M = A.rows();
    N = A.cols();
    cg_iters = cg_iters_;
    rho = rho_;

    // H = |A'*A|^2  (element-wise squared absolute value)
    CMatrix AtA = A.adjoint() * A;
    H = AtA.cwiseAbs2().real();

    // D: difference operator (N-1 x N)
    D.resize(N-1, N);
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < N-1; i++) {
        triplets.emplace_back(i, i,     -1.0);
        triplets.emplace_back(i, i + 1,  1.0);
    }
    D.setFromTriplets(triplets.begin(), triplets.end());

    // DtD = D'*D
    SparseMat DtD = D.transpose() * D;

    // M = H + rho*(I + D'*D)
    M_mat = H;
    M_mat.diagonal().array() += rho;  // I
    M_mat += rho * DtD;               // D'*D (D is already sparse)

    // Allocate workspace
    x.resize(N);  z1.resize(N);  z2.resize(N-1);
    u1.resize(N); u2.resize(N-1);
    cv.resize(N);
    bk.resize(N); r.resize(N); p.resize(N); Mp.resize(N);
    v1.resize(N); v2.resize(N-1);
}

Array ADMMSolver::solve(const CMatrix& C, int it, double tv) {
    // c_vec[i] = real(A_i' * C * A_i)
    for (int i = 0; i < N; i++) {
        CArray col = A.col(i).array();
        cv(i) = (col.conjugate() * (C * col.matrix())).real()(0,0);
    }

    // Reset state
    x.setZero();  z1.setZero();  z2.setZero();
    u1.setZero(); u2.setZero();

    double rho_tv = tv / rho;

    for (int k = 1; k <= it; k++) {
        double tk = (k > 1) ? std::pow(0.5, k) : 1.0;

        // bk = cv + rho*(z1 - u1) + rho*D'*(z2 - u2)
        bk = cv + rho * (z1 - u1);
        bk += rho * (D.transpose() * (z2 - u2).matrix()).array();

        // CG: solve M * x = bk, warm-start from previous x
        r = bk - (M_mat * x.matrix()).array();
        p = r;
        double rs_old = r.matrix().squaredNorm();

        for (int j = 0; j < cg_iters; j++) {
            Mp = (M_mat * p.matrix()).array();
            double alpha = rs_old / p.matrix().dot(Mp.matrix());
            x += alpha * p;
            r -= alpha * Mp;
            double rs_new = r.matrix().squaredNorm();
            if (std::sqrt(rs_new) < tk) break;
            p = r + (rs_new / rs_old) * p;
            rs_old = rs_new;
        }

        // z1 = max(0, x + u1) - soft threshold is identity when tv/rho = 0
        v1 = x + u1;
        z1 = v1.max(0.0);

        // z2 = soft_threshold(D*x + u2, tv/rho)
        v2 = (D * x.matrix()).array() + u2;
        z2 = v2.sign() * (v2.abs() - rho_tv).max(0.0);

        // Dual update
        u1 += x - z1;
        u2 += (D * x.matrix()).array() - z2;
    }

    return z1;
}

} // namespace lam
