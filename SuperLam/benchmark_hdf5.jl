# ==============================================================================
# BENCHMARK: CHAMBOLLE-POCK & SUBGRADIENT ON SYNTHETIC HDF5 DATA
# ==============================================================================

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using SuperLam
using HDF5
using Plots
using LinearAlgebra
using SparseArrays

# Load precomputed assets (grid R, steering matrix A, neighbors)
include("assets/assets_loader.jl")

# ---------------------------------------------------------------------------
# Compute per-band Hessian on-the-fly (matches synthetic data generation)
# Synthetic data uses log-spaced mel frequencies: 50-4500 Hz, 15 bands
# ---------------------------------------------------------------------------
function mel_frequencies(n_mels, fmin, fmax)
    mels = range(0, stop=2595*log10(1 + fmax/700), length=n_mels+2)
    mels = mels[2:end-1]
    return 700 .*(10.0 .^(mels/2595) .- 1)
end

freq_bands = mel_frequencies(15, 50, 4500)
println("  Frequency bands: $(round.(Int, freq_bands)) Hz")

# Eigenmike coordinates (same as assets)
function get_xyz_em()
    _EIGENMIKE_ = Dict(
        "1"=>[69,0,0.042], "2"=>[90,32,0.042], "3"=>[111,0,0.042], "4"=>[90,328,0.042], "5"=>[32,0,0.042],
        "6"=>[55,45,0.042], "7"=>[90,69,0.042], "8"=>[125,45,0.042], "9"=>[148,0,0.042], "10"=>[125,315,0.042],
        "11"=>[90,291,0.042], "12"=>[55,315,0.042], "13"=>[21,91,0.042], "14"=>[58,90,0.042], "15"=>[121,90,0.042],
        "16"=>[159,89,0.042], "17"=>[69,180,0.042], "18"=>[90,212,0.042], "19"=>[111,180,0.042], "20"=>[90,148,0.042],
        "21"=>[32,180,0.042], "22"=>[55,225,0.042], "23"=>[90,249,0.042], "24"=>[125,225,0.042], "25"=>[148,180,0.042],
        "26"=>[125,135,0.042], "27"=>[90,111,0.042], "28"=>[55,135,0.042], "29"=>[21,269,0.042], "30"=>[58,270,0.042],
        "31"=>[122,270,0.042], "32"=>[159,271,0.042])
    xyz = zeros(3, 32)
    for (ch, v) in _EIGENMIKE_
        idx = parse(Int, ch)
        colat, lon, r = deg2rad(v[1]), deg2rad(v[2]), v[3]
        xyz[1, idx] = r * sin(colat) * cos(lon)
        xyz[2, idx] = r * sin(colat) * sin(lon)
        xyz[3, idx] = r * cos(colat)
    end
    return xyz
end

# Compute Hessian for band 10 (higher freq → better conditioning)
band_target = 10
freq = freq_bands[band_target]
c_speed = 343.0
wl = c_speed / freq
scale = 2π / wl
xyz = get_xyz_em()
A_band = exp.(-1im * scale * (xyz') * R)
H_band = abs.(A_band' * A_band).^2

println("  Band $band_target: $(round(Int, freq)) Hz, wl=$(round(wl, digits=4))m")
println("  H_band: $(size(H_band))")

# ---------------------------------------------------------------------------
# Load synthetic dataset
# ---------------------------------------------------------------------------
hdf_path = "../dataset/synthetic_data/synthetic.hdf"
println("Loading: $hdf_path")

em32_all = h5read(hdf_path, "em32")   # (n_samples, n_bands, 32, 32)
apgd_all = h5read(hdf_path, "apgd")   # (n_samples, n_bands, N_px)
dur_all  = h5read(hdf_path, "dur")    # (n_samples,)

println("  em32 shape: $(size(em32_all))")
println("  apgd shape: $(size(apgd_all))")
println("  dur shape:  $(size(dur_all))")

# HDF5 written by Python (row-major), read by Julia (column-major) → dimensions reversed
# Python wrote: (n_samples, n_bands, N_mics, N_mics)
# Julia reads:  (N_mics, N_mics, n_bands, n_samples)
n_samples = size(em32_all, 4)
n_bands = size(em32_all, 3)
N_mics = size(em32_all, 1)
N_px = size(apgd_all, 1)

println("  Parsed: Samples=$n_samples, Bands=$n_bands, Mics=$N_mics, Pixels=$N_px")
println("  Grid R: $(size(R))")
println("  Hessian H: $(size(H_em32))")

# ---------------------------------------------------------------------------
# Solver parameters
# ---------------------------------------------------------------------------
λ_1 = 0.0
λ_tv = 0.01
cp_max_iters = 1000
sg_max_iters = 2000

# ---------------------------------------------------------------------------
# Run on first few samples, band 3
# ---------------------------------------------------------------------------
band = 3
n_test = min(4, n_samples)

results = Dict()

for i in 1:n_test
    println("\n--- Sample $i (Band $band) ---")
    
    # Extract covariance matrix for this sample/band
    # Julia reads Python HDF5 with reversed dims: (N_mics, N_mics, n_bands, n_samples)
    C = em32_all[:, :, band, i]
    x_true = apgd_all[:, band, i]
    
    # Chambolle-Pock
    println("  Running Chambolle-Pock...")
    x_cp = solve_chambolle_pock(A_band, C; max_iters=cp_max_iters, λ_1=λ_1, λ_tv=λ_tv)
    
    # Subgradient
    println("  Running Subgradient...")
    x_sg = solve_subgradient(A_band, C; max_iters=sg_max_iters, λ_1=λ_1, λ_tv=λ_tv, α=k->10.0 * 0.95^k)
    
    results[i] = (x_true=x_true, x_cp=x_cp, x_sg=x_sg)
    
    # Print metrics
    err_cp = norm(x_cp - x_true) / norm(x_true)
    err_sg = norm(x_sg - x_true) / norm(x_true)
    println("  CP relative error: $(round(err_cp, digits=4))")
    println("  SG relative error: $(round(err_sg, digits=4))")
end

# ---------------------------------------------------------------------------
# Plot true vs recreated
# ---------------------------------------------------------------------------
for i in 1:n_test
    r = results[i]
    
    p = plot(r.x_true, label="true", linewidth=2, color=:blue,
             title="Sample $i (Band $band_target, $(round(Int,freq))Hz, Dur=$(round(dur_all[i], digits=3))s)")
    plot!(p, r.x_cp, label="chambolle-pock", linewidth=1.5, color=:green)
    plot!(p, r.x_sg, label="subgradient", linewidth=1.5, color=:purple)
    xlabel!(p, "Grid Index")
    ylabel!(p, "Intensity")
    
    savefig(p, "benchmark_sample$(lpad(i, 2, "0"))_band$(lpad(band_target, 2, "0")).png")
    println("Saved: benchmark_sample$(lpad(i, 2, "0"))_band$(lpad(band_target, 2, "0")).png")
end

# ---------------------------------------------------------------------------
# Summary grid
# ---------------------------------------------------------------------------
p_grid = plot(layout=(n_test, 1), size=(800, 300 * n_test))
for i in 1:n_test
    r = results[i]
    plot!(p_grid[i], r.x_true, label="true", linewidth=2, color=:blue)
    plot!(p_grid[i], r.x_cp, label=i==1 ? "chambolle-pock" : "", linewidth=1.5, color=:green)
    plot!(p_grid[i], r.x_sg, label=i==1 ? "subgradient" : "", linewidth=1.5, color=:purple)
    title!(p_grid[i], "Sample $i")
end
savefig(p_grid, "benchmark_summary_band$(lpad(band_target, 2, "0")).png")
println("\nSaved: benchmark_summary_band$(lpad(band_target, 2, "0")).png")
