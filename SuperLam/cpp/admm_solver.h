#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include <vector>

namespace lam {

using Complex = std::complex<double>;
using Array = Eigen::ArrayXd;
using CArray = Eigen::ArrayXcd;
using Matrix = Eigen::MatrixXd;
using CMatrix = Eigen::MatrixXcd;
using SparseMat = Eigen::SparseMatrix<double>;

struct ADMMSolver {
    int N;                // number of pixels (grid points)
    int M;                // number of mics (32)
    int cg_iters;         // inner CG iterations (default 10)
    double rho;           // ADMM penalty parameter
    CMatrix A;            // steering matrix (M x N), complex

    // Precomputed matrices
    Matrix H;             // |A'*A|^2  (N x N)
    SparseMat D;          // difference operator (N-1 x N)
    Matrix M_mat;         // H + rho*(I + D'*D)  (N x N)

    // Workspace (reused per call)
    Array x, z1, z2, u1, u2;
    Array cv;             // c_vec = diag(A' * C * A)
    Array bk, r, p, Mp;
    Array v1, v2;

    ADMMSolver() = default;

    // Initialize from steering matrix A
    void init(const CMatrix& A_, int cg_iters_ = 10, double rho_ = 100.0);

    // Solve one frame: returns acoustic intensity map z1
    Array solve(const CMatrix& C, int it = 10, double tv = 1.0);
};

} // namespace lam
