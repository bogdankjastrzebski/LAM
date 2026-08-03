function solve_subgradient(A, C; max_iters=5000, λ_1=0.1, λ_tv=0.5, α=k->1/k)
    N = size(A, 2)

    H = abs.(A' * A).^2
    c_vec = zeros(Float64, N)
    for i in 1:N
        c_vec[i] = real(A[:, i]' * C * A[:, i])
    end

    x = zeros(Float64, N)

    for k in 1:max_iters
        g_smooth = H * x .- c_vec

        g_tv = zeros(Float64, N)
        for i in 1:N-1
            s = sign(x[i] - x[i+1])
            v = 1.0
            g_tv[i] += s * v
            g_tv[i+1] -= s * v
        end

        g_l1 = ones(Float64, N)
        d = g_smooth .+ (λ_tv .* g_tv) .+ (λ_1 .* g_l1)
        x = max.(0.0, x .- α(k) .* d / max(norm(d), 1e-10))
    end

    return x
end
