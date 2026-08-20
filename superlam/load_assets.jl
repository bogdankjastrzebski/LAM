"""
load_assets.jl - Load precomputed solver assets from .npy files.

Loads steering matrix A, spherical grid R, neighbor indices, and Hessian H
from the assets/ directory. Returns an Assets struct ready for the solver.

Usage:
    include("load_assets.jl")
    assets = load_assets()
    # or with custom path:
    assets = load_assets(joinpath(@__DIR__, "assets"))
"""

using NPZ

struct Assets
    R::Matrix{Float64}          # (3, N_px) spherical grid
    neighbors::Matrix{Int64}    # (N_px, k) k-NN indices
    A_em32::Matrix{ComplexF64}  # (32, N_px) steering matrix
    A_mic::Matrix{ComplexF64}   # (4, N_px) 4-ch steering matrix
    H_em32::Matrix{Float64}     # (N_px, N_px) Hessian |A'A|^2
    H_mic::Matrix{Float64}      # (N_px, N_px) Hessian (4-ch)
    N_px::Int
    N_mics_em32::Int
    N_mics_mic::Int
    k_neighbors::Int
end

function load_assets(assets_dir::String=joinpath(@__DIR__, "assets"))
    println("Loading assets from: ", assets_dir)
    
    R = npzread(joinpath(assets_dir, "R.npy"))
    neighbors = npzread(joinpath(assets_dir, "neighbors.npy"))
    A_em32 = npzread(joinpath(assets_dir, "A_em32.npy"))
    A_mic = npzread(joinpath(assets_dir, "A_mic.npy"))
    H_em32 = npzread(joinpath(assets_dir, "H_em32.npy"))
    H_mic = npzread(joinpath(assets_dir, "H_mic.npy"))
    
    N_px = size(R, 2)
    
    println("  R:          $(size(R)) $(eltype(R))")
    println("  neighbors:  $(size(neighbors)) $(eltype(neighbors))")
    println("  A_em32:     $(size(A_em32)) $(eltype(A_em32))")
    println("  A_mic:      $(size(A_mic)) $(eltype(A_mic))")
    println("  H_em32:     $(size(H_em32)) $(eltype(H_em32))")
    println("  H_mic:      $(size(H_mic)) $(eltype(H_mic))")
    
    Assets(R, neighbors, A_em32, A_mic, H_em32, H_mic,
           N_px, size(A_em32, 1), size(A_mic, 1), size(neighbors, 2))
end
