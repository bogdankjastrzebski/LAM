"""
    solve_admm(A, C; max_iters=500, λ_1=0.1, λ_tv=0.5, ρ=1.0, tol=1e-4)

ADMM solver for graph TV regularized acoustic mapping.

Splits the problem into:
  x-update: quadratic data fidelity + ADMM penalty (solved via CG)
  z-update: closed-form proximal operators for L1, TV, and non-negativity

# Arguments
- `A`: Steering matrix (N_mics, N_spatial)
- `C`: Covariance matrix (N_mics, N_mics)

# Keyword Arguments
- `max_iters`: Maximum ADMM iterations
- `λ_1`: L1 regularization strength
- `λ_tv`: Total variation regularization strength
- `ρ`: ADMM penalty parameter
- `tol`: Convergence tolerance on primal residual

# Returns
- `x`: Reconstructed acoustic intensity map (N_spatial,)
"""
function solve_admm(A, C; max_iters=500, λ_1=0.1, λ_tv=0.5, ρ=1.0, outer_tol=1e-4, cg_tol_factor=0.5)
    N = size(A, 2)

    # 1. PRECOMPUTATION
    # ---------------------------------------------------------
    H = abs.(A' * A).^2
    c_vec = zeros(Float64, N)
    for i in 1:N
        c_vec[i] = real(A[:, i]' * C * A[:, i])
    end

    D = spdiagm(0 => -ones(N), 1 => ones(N-1))[1:N-1, :]
    DtD = D' * D

    # The Strongly Convex ADMM Hessian
    M = H + ρ * (I + DtD)

    # 2. STATE INITIALIZATION
    # ---------------------------------------------------------
    x = zeros(Float64, N)
    z1 = zeros(Float64, N)   # Split for x >= 0 and L1
    z2 = zeros(Float64, N-1) # Split for TV edges
    u1 = zeros(Float64, N)   # Dual for x = z1
    u2 = zeros(Float64, N-1) # Dual for Dx = z2

    # CG workspace caches
    r_cache = zeros(Float64, N)
    p_cache = zeros(Float64, N)
    Mp_cache = zeros(Float64, N)
    b_k = zeros(Float64, N)
    
    primal_res_old = 1.0 

    println("Starting ADMM (cg_tol_factor = $cg_tol_factor)...")

    # 3. THE ALTERNATING LOOP
    # ---------------------------------------------------------
    s = 0
    for k in 1:max_iters
        # --- DYNAMIC TOLERANCE SCALING ---
        if k == 1
            tol_k = 1.0 
        else
            # Scale tolerance relative to previous primal residual, forced by user hyperparameter
            # tol_k = max((cg_tol_factor * primal_res_old)^k, 1e-7)
            tol_k = cg_tol_factor^k
        end

        # --- STEP A: X-UPDATE (Inner CG) ---
        @. b_k = c_vec + ρ * (z1 - u1)
        b_k .+= ρ .* (D' * (z2 .- u2))

        inner_steps = solve_cg_inexact!(x, M, b_k, 
                                        r=r_cache, p=p_cache, Hp=Mp_cache, 
                                        tol=tol_k, max_iters=2)

        # --- STEP B: Z-UPDATE (Proximal Operators) ---
        # z1: Soft threshold for L1, then clamp at 0 for non-negativity
        v1 = x .+ u1
        z1_new = max.(0.0, sign.(v1) .* max.(abs.(v1) .- λ_1 / ρ, 0.0))

        # z2: Soft threshold for TV
        v2 = D * x .+ u2
        z2_new = sign.(v2) .* max.(abs.(v2) .- λ_tv / ρ, 0.0)

        # --- STEP C: DUAL UPDATE ---
        u1 .= u1 .+ x .- z1_new
        u2 .= u2 .+ (D * x) .- z2_new

        # --- CONVERGENCE CHECK ---
        primal_res = max(norm(x - z1_new), norm(D * x - z2_new))
        primal_res_old = primal_res 
        
        if primal_res < outer_tol
            println("ADMM converged at iter $k | Res: $(round(primal_res, digits=6)) | CG Steps: $inner_steps")
            z1 = z1_new
            z2 = z2_new
            break
        end

        z1 = z1_new
        z2 = z2_new

        if k % 1 == 0 || k == 1
            println("  Iter $k | CG Steps: $(lpad(inner_steps, 2)) | CG Tol: $(round(tol_k, digits=6)) | Primal residual: $(round(primal_res, digits=6))")
        end
    end

    # Return z1 to guarantee exact output bounds (strictly non-negative, exact zeroes)
    return z1
end
