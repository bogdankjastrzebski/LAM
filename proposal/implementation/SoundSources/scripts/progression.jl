using SoundSources, Flux, Zygote, LinearAlgebra, Plots

assets = load_assets_4mic()
A = assets.A; H_mat = assets.H; N = size(A, 2)
R_ang = cartesian_to_angular(assets.R)
az = sort(R_ang[1,:]); el_sort_idx = sortperm(R_ang[1,:])

# Generate synthetic ground truth (2 sources + noise)
x_true = zeros(N)
sources = [(az=π/2, el=0.2, amp=5.0), (az=π, el=0.1, amp=3.0)]
for s in sources
    for j in 1:N
        daz = abs(R_ang[1,j] - s.az)
        daz = min(daz, 2π - daz)
        dist2 = daz^2 + (R_ang[2,j] - s.el)^2
        x_true[j] += s.amp * exp(-dist2 / (2 * 0.12^2))
    end
end

C = A * Diagonal(x_true) * A'
C += 0.03 * (randn(ComplexF64, 4, 4) |> m -> m * m')

c_vec = zeros(N)
for i in 1:N; c_vec[i] = real(A[:,i]' * C * A[:,i]); end

# ADMM baseline
println("ADMM...")
x_admm = solve_admm(A, C; max_iters=200, λ_1=0.1, λ_tv=0.5, ρ=1.0, cg_tol_factor=0.5)

# Gaussian fitting with snapshots
K = 8; v = init_gaussians(K); opt = Flux.Optimise.Adam(5e-3)
snap_iter = [1, 10, 50, 100, 250, 500, 1000]
snapshots = Vector{Float64}[]

for iter in 1:1000
    gs = Zygote.gradient(v) do vp
        xr = render_from_flat(vp, R_ang)
        return 0.5 * dot(xr, H_mat * xr) - dot(c_vec, xr)
    end
    Flux.Optimise.update!(opt, v, gs[1])
    if iter in snap_iter
        push!(snapshots, copy(render_from_flat(v, R_ang)))
    end
end

x_gauss = render_from_flat(v, R_ang)

sort_idx = sortperm(R_ang[1,:])
xs_true = x_true[sort_idx]
xs_admm = x_admm[sort_idx]
xs_gauss = x_gauss[sort_idx]
xs_snaps = [s[sort_idx] for s in snapshots]
az_sorted = R_ang[1,sort_idx]

plots_list = []
push!(plots_list, plot(az_sorted, xs_true, title="Ground Truth", legend=false, linewidth=2))
push!(plots_list, plot(az_sorted, xs_admm, title="ADMM (484 params)", legend=false, linewidth=1.5))

for (i, s) in enumerate(snapshots)
    label = "iter $(snap_iter[i])"
    push!(plots_list, plot(az_sorted, xs_snaps[i], title=label, legend=false, linewidth=1.5))
end

push!(plots_list, plot(az_sorted, xs_gauss, title="Gauss final (1000)", legend=false, linewidth=2))

n = length(plots_list)
plot(plots_list..., layout=(n, 1), size=(700, 150*n))
savefig("progression.png")
println("Saved progression.png ($n panels)")

# Also print where Gaussians converged
θ = extract_gaussians(v)
println("\nFitted Gaussians (μ in radians, az=π/2=1.57, az=π=3.14):")
for (k, g) in enumerate(θ)
    if g.π > 0.1
        println("  k=$k: az=$(round(g.μ[1], digits=2))  el=$(round(g.μ[2], digits=2))  σ=$(round(g.σ, digits=3))  π=$(round(g.π, digits=3))")
    end
end
