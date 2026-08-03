function solve_cg_inexact!(x, H, b; r, p, Hp, tol=1e-5, max_iters=500)
    mul!(Hp, H, x)
    @. r = b - Hp
    copyto!(p, r)
    
    rs_old = dot(r, r)
    
    if sqrt(rs_old) < tol
        return 0
    end
    
    inner_iters = 0
    for j in 1:max_iters
        inner_iters += 1
        
        mul!(Hp, H, p)
        
        alpha = rs_old / dot(p, Hp)
        
        @. x += alpha * p
        @. r -= alpha * Hp
        
        rs_new = dot(r, r)
        
        if sqrt(rs_new) < tol
            break
        end
        
        @. p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    end
    
    return inner_iters
end
