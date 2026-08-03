using LinearAlgebra
using Flux

function loss_and_gradient_wishart!(grad, v, mic_positions, wavelength, C_obs)
    K = length(v) ÷ 4
    n_mics = size(mic_positions, 2)

    σ²_noise = 1e-2  # noise floor

    C_pred = σ²_noise * Matrix{Float64}(I, n_mics, n_mics) |> complex
    a_cache = Vector{Vector{ComplexF64}}(undef, K)
    for k in 1:K
        az = v[4k-3]; el = v[4k-2]
        pi = exp(v[4k]) + 1e-8
        s = source_direction(az, el)
        a = steering_vector(mic_positions, wavelength, s)
        a_cache[k] = a
        C_pred += pi * a * a'
    end

    # Wishart negative log-likelihood (up to constants):
    # L = log det(C_pred) + tr(C_pred^{-1} * C_obs)
    Cinv = inv(C_pred)
    tr_term = real(tr(Cinv * C_obs))
    ld = real(log(det(C_pred)))
    loss = ld + tr_term

    fill!(grad, 0.0)
    for k in 1:K
        az = v[4k-3]; el = v[4k-2]
        pi = exp(v[4k]) + 1e-8
        a = a_cache[k]
        da_az = dsteering_daz(mic_positions, wavelength, az, el)
        da_el = dsteering_del(mic_positions, wavelength, az, el)

        # ∂/∂π_k
        # ∂L/∂π_k = tr(Cinv * a*a') - tr(Cinv * C_obs * Cinv * a*a')
        dC_dpi = a * a'
        grad_pi = real(tr(Cinv * dC_dpi) - tr(Cinv * C_obs * Cinv * dC_dpi))
        grad[4k] += grad_pi * pi

        # ∂/∂μ_k: ∂L/∂μ = tr(Cinv * ∂C/∂μ) - tr(Cinv * C_obs * Cinv * ∂C/∂μ)
        # ∂C/∂az = pi * (da_az*a' + a*da_az')
        dC_az = pi * (da_az * a' + a * da_az')
        grad[4k-3] += real(tr(Cinv * dC_az) - tr(Cinv * C_obs * Cinv * dC_az))

        dC_el = pi * (da_el * a' + a * da_el')
        grad[4k-2] += real(tr(Cinv * dC_el) - tr(Cinv * C_obs * Cinv * dC_el))
    end

    return loss
end

function solve_gaussians_wishart!(v, mic_positions, wavelength, C_obs; max_iters=10, lr=1e-3)
    grad = zeros(length(v))
    opt = Flux.Optimise.Adam(lr)
    for _ in 1:max_iters
        loss_and_gradient_wishart!(grad, v, mic_positions, wavelength, C_obs)
        Flux.Optimise.update!(opt, v, grad)
        for k in 1:(length(v)÷4)
            if v[4k-2] < -0.3; v[4k-2] = -0.3; end
        end
    end
    return v
end
