using SoundSources
using Plots
using LinearAlgebra
using ProgressMeter

assets = load_assets_4mic()
A = assets.A
R_grid = assets.R
H = assets.H
N = size(A, 2)
R_ang = cartesian_to_angular(R_grid)

function generate_synthetic_sources(R_ang)
    N = size(R_ang, 2)
    x_true = zeros(N)
    sources = [
        (az=π/2, el=0.2, amp=5.0),
        (az=π,   el=0.1, amp=3.0),
    ]
    for s in sources
        for j in 1:N
            az_j, el_j = R_ang[1, j], R_ang[2, j]
            daz = abs(az_j - s.az)
            daz = min(daz, 2π - daz)
            dist2 = daz^2 + (el_j - s.el)^2
            x_true[j] += s.amp * exp(-dist2 / (2 * 0.15^2))
        end
    end
    return x_true
end

x_true = generate_synthetic_sources(R_ang)

C = A * Diagonal(x_true) * A'
noise_std = 0.05
C_noisy = C + noise_std * (randn(ComplexF64, 4, 4) |> m -> m * m')

c_vec = zeros(N)
for i in 1:N
    c_vec[i] = real(A[:, i]' * C_noisy * A[:, i])
end

println("admm:")
@time x_admm = solve_admm(A, C_noisy; max_iters=200, λ_1=0.1, λ_tv=0.5, ρ=1.0, cg_tol_factor=0.5)

println("gaussians:")
@time x_gauss, θ_gauss = solve_gaussians(A, C_noisy; K=8, max_iters=3000, lr=5e-3)

println()
println("admm loss:     $(round(objective_value(x_admm, H, c_vec), digits=2))")
println("gauss loss:    $(round(objective_value(x_gauss, H, c_vec), digits=2))")
println("true loss:     $(round(objective_value(x_true, H, c_vec), digits=2))")

az = R_ang[1, :]
p1 = plot(az, x_true, title="True Sources", xlabel="azimuth", ylabel="intensity", legend=false)
p2 = plot(az, x_admm, title="ADMM (484 vars)", xlabel="azimuth", legend=false)
p3 = plot(az, x_gauss, title="Gaussians K=8 (32 vars)", xlabel="azimuth", legend=false)

plot(p1, p2, p3, layout=(3, 1), size=(800, 600))
savefig("comparison_output.png")
println("saved comparison_output.png")

for (k, g) in enumerate(θ_gauss)
    println("  k=$k: az=$(round(g.μ[1], digits=2)) el=$(round(g.μ[2], digits=2))  σ=$(round(g.σ, digits=3))  π=$(round(g.π, digits=3))")
end
