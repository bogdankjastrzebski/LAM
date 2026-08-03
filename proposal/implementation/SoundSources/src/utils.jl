function load_assets_4mic()
    return (R=R, A=A_mic, H=H_mic, neighbors=neighbors)
end

function cartesian_to_angular(R_cart)
    N = size(R_cart, 2)
    R_ang = zeros(2, N)
    for j in 1:N
        x, y, z = R_cart[1, j], R_cart[2, j], R_cart[3, j]
        az = atan(y, x)
        az = ifelse(az < 0, az + 2π, az)
        el = asin(z)
        R_ang[1, j] = az
        R_ang[2, j] = el
    end
    return R_ang
end

function objective_value(x, H, c_vec, λ_1=0.0, λ_tv=0.0)
    N = length(x)
    loss = 0.5 * dot(x, H * x) - dot(c_vec, x)
    if λ_1 > 0
        loss += λ_1 * sum(abs.(x))
    end
    if λ_tv > 0
        D = spdiagm(0 => -ones(N), 1 => ones(N-1))[1:N-1, :]
        loss += λ_tv * sum(abs.(D * x))
    end
    return loss
end
