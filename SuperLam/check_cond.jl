include("assets/assets_loader.jl")
using LinearAlgebra

H = Float64.(H_em32)
ev = sort(real(eigvals(H)))

println("Eigenvalues of H (484 total):")
println("  min:  $(ev[1])")
println("  max:  $(ev[end])")
println("  gap to next: [1→2] = $(ev[2]-ev[1])")
println("  # < 1e-10: $(sum(ev .< 1e-10))")
println()

# Show bottom eigenvalues
for k in [1,2,3,4,5,10,50,100,150,200,300,400,450]
    println("  ev[$k] = $(ev[k])")
end

# Check rank: number of eigenvalues > 1e-12
tol = 1e-12
println("\nRank estimate: $(sum(ev .> tol)) / 484")

# Matrix A has rank at most 32 (N_mics). 
# A^H A has rank ≤ 32.
# H = |A^H A|.^2 takes element-wise absolute square - this is nonlinear
# and can INCREASE rank beyond 32.

# Let's check the original A^H A rank
println("\nA^H A eigenvalues:")
A = A_em32
AtA = A' * A
ev2 = sort(real(eigvals(Matrix(AtA))))
println("  max: $(ev2[end])")
println("  # > 1e-12: $(sum(ev2 .> 1e-12))")
