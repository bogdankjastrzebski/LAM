# ==============================================================================
# LAM SOLVER COMPARISON - 1D PROTOTYPE
# ==============================================================================

include("src/LAM.jl")
using .LAM: solve_amortized_cg, solve_chambolle_pock, solve_subgradient, solve_inexact_dual_ascent

using Plots
using LinearAlgebra

# 1. Forward Model Generator (1D Array Simulation)
function generate_acoustic_problem(N_spatial=1000, N_mics=16)
    angles = range(-pi/2, pi/2, length=N_spatial)
    A = zeros(ComplexF64, N_mics, N_spatial)
    for (i, θ) in enumerate(angles)
        for m = 1:N_mics
            A[m, i] = exp(-im * pi * (m-1) * sin(θ))
        end
    end
    x_true = zeros(Float64, N_spatial)
    x_true[200:250] .= 5.0
    x_true[700:800] .= 3.0
    C_true = A * Diagonal(x_true) * A'
    C = C_true + 0.05 * (randn(ComplexF64, N_mics, N_mics) |> m -> m * m')
    return A, C, x_true
end

# ==============================================================================
# EXECUTION
# ==============================================================================

A, C, x_true = generate_acoustic_problem(1000, 16)

# Run the Amortized Solver
x_cg = solve_amortized_cg(A, C, max_iters=100, λ_1=0.0, λ_tv=1000.0)

x_cp = solve_chambolle_pock(A, C; max_iters=10, λ_1=0.0, λ_tv=1000.0)

x_sg = solve_subgradient(A, C; max_iters=100, λ_1=0.0, λ_tv=1000, α=k->10.0 * 0.95^k)

x_da = solve_inexact_dual_ascent(A, C; max_outer_iters=100, λ_tv=0.5, μ=1e-2)

plot(x_true, label="true")
plot!(x_cg, label="cg")
plot!(x_cp, label="chambolle-pock")
plot!(x_sg, label="subgrad")
title!("")
savefig("comparison_16_fast.png")
