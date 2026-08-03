"""
    solve_admm(A, C; max_iters=500, λ_1=0.1, λ_tv=0.5, ρ=1.0, outer_tol=1e-4, cg_tol_factor=0.5)

ADMM solver for graph TV regularized acoustic mapping.
"""
function solve_admm(A, C; max_iters=500, λ_1=0.1, λ_tv=0.5, ρ=1.0, outer_tol=1e-4, cg_tol_factor=0.5)
    N = size(A, 2)

    H = abs.(A' * A).^2
    c_vec = zeros(Float64, N)
    for i in 1:N
        c_vec[i] = real(A[:, i]' * C * A[:, i])
    end

    D = spdiagm(0 => -ones(N), 1 => ones(N-1))[1:N-1, :]
    DtD = D' * D

    M = H + ρ * (I + DtD)

    x = zeros(Float64, N)
    z1 = zeros(Float64, N)
    z2 = zeros(Float64, N-1)
    u1 = zeros(Float64, N)
    u2 = zeros(Float64, N-1)

    r_cache = zeros(Float64, N)
    p_cache = zeros(Float64, N)
    Mp_cache = zeros(Float64, N)
    b_k = zeros(Float64, N)
    
    primal_res_old = 1.0 

    for k in 1:max_iters
        if k == 1
            tol_k = 1.0 
        else
            tol_k = cg_tol_factor^k
        end

        @. b_k = c_vec + ρ * (z1 - u1)
        b_k .+= ρ .* (D' * (z2 .- u2))

        solve_cg_inexact!(x, M, b_k, r=r_cache, p=p_cache, Hp=Mp_cache, tol=tol_k, max_iters=2)

        v1 = x .+ u1
        z1_new = max.(0.0, sign.(v1) .* max.(abs.(v1) .- λ_1 / ρ, 0.0))

        v2 = D * x .+ u2
        z2_new = sign.(v2) .* max.(abs.(v2) .- λ_tv / ρ, 0.0)

        u1 .= u1 .+ x .- z1_new
        u2 .= u2 .+ (D * x) .- z2_new

        primal_res = max(norm(x - z1_new), norm(D * x - z2_new))
        primal_res_old = primal_res 
        
        if primal_res < outer_tol
            z1 = z1_new
            z2 = z2_new
            break
        end

        z1 = z1_new
        z2 = z2_new
    end

    return z1
end
