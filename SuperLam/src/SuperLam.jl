module SuperLam

using LinearAlgebra
using SparseArrays
using JuMP
using OSQP

include("conjugate_gradient.jl")
include("amortized_cg.jl")
include("chambolle_pock.jl")
include("subgradient.jl")
include("dual_ascent.jl")
include("admm.jl")

export solve_cg_inexact!
export solve_amortized_cg
export solve_chambolle_pock
export solve_subgradient
export solve_inexact_dual_ascent
export solve_admm

end # module SuperLam
