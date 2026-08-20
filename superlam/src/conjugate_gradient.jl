using LinearAlgebra
using SparseArrays

function solve_cg_inexact!(x, H, b; r, p, Hp, tol=1e-5, max_iters=500)
    # Initialize residual: r = b - H*x
    mul!(Hp, H, x)
    @. r = b - Hp
    copyto!(p, r)
    
    rs_old = dot(r, r)
    
    # Early exit if we are already within the dynamic tolerance
    if sqrt(rs_old) < tol
        return 0
    end
    
    inner_iters = 0
    for j in 1:max_iters
        inner_iters += 1
        
        # Matrix-vector multiply (the only expensive part)
        mul!(Hp, H, p)
        
        # Step size
        alpha = rs_old / dot(p, Hp)
        
        # Update primal variable and residual
        @. x += alpha * p
        @. r -= alpha * Hp
        
        rs_new = dot(r, r)
        
        # THE EXACT ORACLE STOPPING CRITERION
        if sqrt(rs_new) < tol
            break
        end
        
        # Conjugate direction update
        @. p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    end
    
    return inner_iters
end
