using LinearAlgebra
using Flux

function init_flat(K::Int)
    v = zeros(4K)
    for k in 1:K
        v[4k-3] = 2π * rand()
        v[4k-2] = (rand() - 0.5) * π
        v[4k-1] = log(0.3)
        v[4k]   = log(0.01)
    end
    return v
end

function source_direction(az, el)
    return [cos(el)*cos(az), cos(el)*sin(az), sin(el)]
end

function dsource_daz(az, el)
    return [-cos(el)*sin(az), cos(el)*cos(az), 0.0]
end

function dsource_del(az, el)
    return [-sin(el)*cos(az), -sin(el)*sin(az), cos(el)]
end

function steering_vector(mic_positions, wavelength, s)
    n_mics = size(mic_positions, 2)
    k = 2π / wavelength
    a = zeros(ComplexF64, n_mics)
    for m in 1:n_mics
        phase = k * dot(mic_positions[:, m], s)
        a[m] = exp(-im * phase)
    end
    return a
end

function dsteering_daz(mic_positions, wavelength, az, el)
    n_mics = size(mic_positions, 2)
    k = 2π / wavelength
    s = source_direction(az, el)
    ds = dsource_daz(az, el)
    da = zeros(ComplexF64, n_mics)
    a = steering_vector(mic_positions, wavelength, s)
    for m in 1:n_mics
        da[m] = -im * k * dot(mic_positions[:, m], ds) * a[m]
    end
    return da
end

function dsteering_del(mic_positions, wavelength, az, el)
    n_mics = size(mic_positions, 2)
    k = 2π / wavelength
    s = source_direction(az, el)
    ds = dsource_del(az, el)
    da = zeros(ComplexF64, n_mics)
    a = steering_vector(mic_positions, wavelength, s)
    for m in 1:n_mics
        da[m] = -im * k * dot(mic_positions[:, m], ds) * a[m]
    end
    return da
end

function loss_and_gradient!(grad, v, mic_positions, wavelength, C_obs)
    K = length(v) ÷ 4

    # Forward: predicted covariance
    C_pred = zeros(ComplexF64, 4, 4)
    a_cache = Vector{Vector{ComplexF64}}(undef, K)
    for k in 1:K
        az = v[4k-3]; el = v[4k-2]
        pi = exp(v[4k]) + 1e-8
        s = source_direction(az, el)
        a = steering_vector(mic_positions, wavelength, s)
        a_cache[k] = a
        C_pred += pi * a * a'
    end

    dC = C_pred - C_obs
    loss = real(dot(dC, dC))

    fill!(grad, 0.0)
    for k in 1:K
        az = v[4k-3]; el = v[4k-2]
        pi = exp(v[4k]) + 1e-8
        a = a_cache[k]

        # ∂/∂π_k: 2 Re(aᴴ·dC·a)
        aH_dC_a = a' * dC * a
        grad[4k] = 2 * real(aH_dC_a) * pi  # chain through exp

        # ∂/∂az: 4 Re(aᴴ·dC·∂a/∂az)
        da_az = dsteering_daz(mic_positions, wavelength, az, el)
        grad[4k-3] = 4 * pi * real(a' * dC * da_az)

        # ∂/∂el: 4 Re(aᴴ·dC·∂a/∂el)
        da_el = dsteering_del(mic_positions, wavelength, az, el)
        grad[4k-2] = 4 * pi * real(a' * dC * da_el)

        # σ gradient: not used in this model (spread via beamwidth)
    end

    return loss
end

# Adam solver
function solve_gaussians!(v, mic_positions, wavelength, C_obs; max_iters=10, lr=1e-3)
    grad = zeros(length(v))
    opt = Flux.Optimise.Adam(lr)
    for _ in 1:max_iters
        loss_and_gradient!(grad, v, mic_positions, wavelength, C_obs)
        Flux.Optimise.update!(opt, v, grad)
        for k in 1:(length(v)÷4)
            if v[4k-2] < -0.3; v[4k-2] = -0.3; end
        end
    end
    return v
end

# Gauss-Newton with Levenberg-Marquardt damping
function solve_gaussians_gn!(v, mic_positions, wavelength, C_obs; max_iters=20, λ_init=1.0)
    K = length(v) ÷ 4
    n_params = length(v)
    n_res = 32  # 4×4 complex → 2·16 real

    λ = λ_init
    prev_loss = Inf

    for iter in 1:max_iters
        # Forward: compute residual r ∈ ℝ³²
        C_pred = zeros(ComplexF64, 4, 4)
        a_cache = Vector{Vector{ComplexF64}}(undef, K)
        da_az_cache = Vector{Vector{ComplexF64}}(undef, K)
        da_el_cache = Vector{Vector{ComplexF64}}(undef, K)
        for k in 1:K
            az = v[4k-3]; el = v[4k-2]
            pi = exp(v[4k]) + 1e-8
            s = source_direction(az, el)
            a = steering_vector(mic_positions, wavelength, s)
            a_cache[k] = a
            da_az_cache[k] = dsteering_daz(mic_positions, wavelength, az, el)
            da_el_cache[k] = dsteering_del(mic_positions, wavelength, az, el)
            C_pred += pi * a * a'
        end

        dC = C_pred - C_obs
        loss = real(dot(dC, dC))

        if abs(loss - prev_loss) < 1e-10
            break
        end
        prev_loss = loss

        # Build residual r ∈ ℝ³²
        r = zeros(n_res)
        idx = 1
        for i in 1:4, j in 1:4
            r[idx] = real(dC[i,j]); idx += 1
            r[idx] = imag(dC[i,j]); idx += 1
        end

        # Build Jacobian J ∈ ℝ³²ˣⁿ
        J = zeros(n_res, n_params)
        for k in 1:K
            a = a_cache[k]
            da_az = da_az_cache[k]
            da_el = da_el_cache[k]
            pi = exp(v[4k]) + 1e-8

            # ∂/∂π_k via log-π:  π_k · a·aᴴ
            col = zeros(ComplexF64, 4, 4)
            for i in 1:4, j in 1:4
                col[i,j] = pi * a[i] * conj(a[j])
            end
            col_idx = 1
            for i in 1:4, j in 1:4
                J[col_idx, 4k] = real(col[i,j]); col_idx += 1
                J[col_idx, 4k] = imag(col[i,j]); col_idx += 1
            end

            # ∂/∂az: π_k · (da_az·aᴴ + a·da_azᴴ)
            col = zeros(ComplexF64, 4, 4)
            for i in 1:4, j in 1:4
                col[i,j] = pi * (da_az[i] * conj(a[j]) + a[i] * conj(da_az[j]))
            end
            col_idx = 1
            for i in 1:4, j in 1:4
                J[col_idx, 4k-3] = real(col[i,j]); col_idx += 1
                J[col_idx, 4k-3] = imag(col[i,j]); col_idx += 1
            end

            # ∂/∂el: π_k · (da_el·aᴴ + a·da_elᴴ)
            col = zeros(ComplexF64, 4, 4)
            for i in 1:4, j in 1:4
                col[i,j] = pi * (da_el[i] * conj(a[j]) + a[i] * conj(da_el[j]))
            end
            col_idx = 1
            for i in 1:4, j in 1:4
                J[col_idx, 4k-2] = real(col[i,j]); col_idx += 1
                J[col_idx, 4k-2] = imag(col[i,j]); col_idx += 1
            end

            # σ: not used, zero column
        end

        # Damped Gauss-Newton: Δv = -(JᵀJ + λI)⁻¹ · Jᵀ · r
        JTJ = J' * J
        JTr = J' * r
        delta = -(JTJ + λ * I) \ JTr

        v_new = v + delta
        for k in 1:K
            if v_new[4k-2] < -0.3; v_new[4k-2] = -0.3; end
        end

        C_new = zeros(ComplexF64, 4, 4)
        for k in 1:K
            az_n = v_new[4k-3]; el_n = v_new[4k-2]
            pi_n = exp(v_new[4k]) + 1e-8
            s_n = source_direction(az_n, el_n)
            a_n = steering_vector(mic_positions, wavelength, s_n)
            C_new += pi_n * a_n * a_n'
        end
        dC_new = C_new - C_obs
        loss_new = real(dot(dC_new, dC_new))

        if loss_new < loss
            v .= v_new
            λ *= 0.5
        else
            λ *= 2.0
        end
    end
    return v
end
