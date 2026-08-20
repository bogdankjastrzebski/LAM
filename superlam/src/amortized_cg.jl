"""
    solve_amortized_cg(A, C; max_iters=50, λ_1=0.1, λ_tv=0.5)

Amortized active-set conjugate gradient solver for graph TV regularized
acoustic mapping.

Uses a warm-started OSQP subproblem to find descent directions with
conjugacy and TV constraints, then performs exact line search with
collision detection to maintain non-negativity.

# Arguments
- `A`: Steering matrix (N_mics, N_spatial)
- `C`: Covariance matrix (N_mics, N_mics)

# Keyword Arguments
- `max_iters`: Maximum outer iterations
- `λ_1`: L1 regularization strength
- `λ_tv`: Total variation regularization strength

# Returns
- `x`: Reconstructed acoustic intensity map (N_spatial,)
"""
function solve_amortized_cg(A, C; max_iters=50, λ_1=0.1, λ_tv=0.5)
    N = size(A, 2)

    H = abs.(A' * A).^2
    c_vec = zeros(Float64, N)
    for i in 1:N
        c_vec[i] = real(A[:, i]' * C * A[:, i])
    end

    x = zeros(Float64, N)
    d_prev = zeros(Float64, N)

    model = Model(OSQP.Optimizer)
    set_silent(model)

    @variable(model, -1 <= d[1:N] <= 1)
    @variable(model, u[1:N-1] >= 0)

    @constraint(model, tv_pos[i=1:N-1], d[i] - d[i+1] - u[i] <= 0)
    @constraint(model, tv_neg[i=1:N-1], -(d[i] - d[i+1]) - u[i] <= 0)

    @constraint(model, conj, sum(1.0 * d[i] for i in 1:N) == 0)

    println("Starting Amortized CG...")

    for k in 1:max_iters
        g = H * x - c_vec

        y = H * d_prev
        y .+= 1e-12

        for i in 1:N
            set_normalized_coefficient(conj, d[i], y[i])
        end

        for i in 1:N
            if x[i] <= 1e-6 && g[i] > 0
                set_lower_bound(d[i], 0.0)
            else
                set_lower_bound(d[i], -1.0)
            end
        end

        @objective(model, Min, sum((g[i] + λ_1) * d[i] for i in 1:N) + λ_tv * sum(u))

        optimize!(model)

        d_k = value.(d)

        if norm(d_k) < 1e-4
            println("Converged perfectly at iteration $k.")
            break
        end

        curvature = dot(d_k, H * d_k)
        alpha_exact = -dot(g .+ λ_1, d_k) / (curvature + 1e-8)

        alpha = alpha_exact
        for i in 1:N
            if d_k[i] < 0
                alpha = min(alpha, -x[i] / d_k[i])
            end
        end

        x = x .+ alpha .* d_k
        d_prev = d_k
    end

    return x
end
