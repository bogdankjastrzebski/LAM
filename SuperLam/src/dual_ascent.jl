function solve_inexact_dual_ascent(A, C; max_outer_iters=100, λ_tv=0.5, μ=1e-2, α=0.1)
    N = size(A, 2)
    
    # 1. PRECOMPUTATION
    H_μ = abs.(A' * A).^2 + μ * I
    c_vec = zeros(Float64, N)
    for i in 1:N
        c_vec[i] = real(A[:, i]' * C * A[:, i])
    end
    
    D = spdiagm(0 => -ones(N), 1 => ones(N-1))[1:N-1, :]
    
    # Exact constants for the stopping rule
    norm_D = 2.0                # For a 1D difference matrix, the spectral norm is exactly 2.0
    L_y = (norm_D^2) / μ        # Lipschitz constant of the dual objective
    outer_step = α / L_y      # Perfect step size
    δ = 0.5                     # We demand the error is strictly less than 50% of the gradient
    
    # State Initialization
    x = zeros(Float64, N)
    y = zeros(Float64, N-1)
    b_k = zeros(Float64, N)
    grad_y = zeros(Float64, N-1)
    
    # CG Caches
    r_cache, p_cache, Hp_cache = zeros(N), zeros(N), zeros(N)
    
    println("Starting μ-Driven Inexact Dual Ascent...")
    
    for k in 1:max_outer_iters
        @show y[1:10]
        mul!(b_k, D', y)
        @. b_k = c_vec - b_k
        
        # ---------------------------------------------------------
        # THE EXACT STOPPING RULE
        # ---------------------------------------------------------
        if k == 1
            # On iteration 1, we don't have a previous gradient, so we use a loose fallback
            tol_k = 1.0 
        else
            # Calculate the explicit threshold using μ and the previous gradient magnitude
            grad_norm = norm(grad_y)
            tol_k = (μ * δ / norm_D) * grad_norm
            
            # Floor the tolerance to hardware epsilon limits to prevent infinite CG loops
            # when we reach the absolute top of the saddle point
            tol_k = max(tol_k, 1e-7)
        end
        
        # INNER MIN
        inner_steps = solve_cg_inexact!(x, H_μ, b_k, 
                                        r=r_cache, p=p_cache, Hp=Hp_cache, 
                                        tol=tol_k)
        
        # OUTER MAX
        mul!(grad_y, D, x)
        @. y = clamp(y + outer_step * grad_y, -λ_tv, λ_tv)
        
        if k % 10 == 0 || k == 1
            println("Outer: $k | CG Steps: $(lpad(inner_steps, 2)) | Gradient Norm: $(round(norm(grad_y), digits=4)) | CG Tol: $(round(tol_k, digits=6))")
        end
    end
    
    return x, y
end
