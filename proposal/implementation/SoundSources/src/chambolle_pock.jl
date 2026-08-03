function solve_chambolle_pock(A, C; max_iters=1000, λ_1=0.1, λ_tv=0.5)
    N = size(A, 2)

    H = abs.(A' * A).^2
    c_vec = zeros(Float64, N)
    for i in 1:N
        c_vec[i] = real(A[:, i]' * C * A[:, i])
    end

    L = opnorm(H)
    τ = 1.0 / L
    σ = 0.9 / (4.0 * τ)

    x = zeros(Float64, N)
    x_bar = zeros(Float64, N)
    y = zeros(Float64, N-1)

    D = spdiagm(0 => -ones(N), 1 => ones(N-1))[1:N-1, :]

    for _ in 1:max_iters
        y_new = y .+ σ .* (D * x_bar)
        y = clamp.(y_new, -λ_tv, λ_tv)

        x_old = copy(x)
        g_smooth = H * x .- c_vec
        v = x .- τ .* (g_smooth .+ D' * y)
        x = max.(0.0, v .- τ * λ_1)
        x_bar = 2.0 .* x .- x_old
    end

    return x
end
