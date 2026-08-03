module SoundSources

using LinearAlgebra
using SparseArrays

include("../assets/assets_loader.jl")

include("utils.jl")
include("conjugate_gradient.jl")
include("admm.jl")
include("chambolle_pock.jl")
include("subgradient.jl")
include("gaussian_tracker.jl")
include("gaussian_tracker_wishart.jl")

export solve_cg_inexact!
export solve_admm
export solve_chambolle_pock
export solve_subgradient
export solve_gaussians!
export solve_gaussians_gn!
export solve_gaussians_wishart!
export loss_and_gradient_wishart!
export init_flat
export loss_and_gradient!
export source_direction
export steering_vector
export dsteering_daz
export dsteering_del
export cartesian_to_angular
export objective_value
export load_assets_4mic

end
