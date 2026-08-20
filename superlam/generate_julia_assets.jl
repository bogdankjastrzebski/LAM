"""
generate_julia_assets.jl - Convert .npy assets to native Julia source files.

Reads the .npy files from assets/ and writes them as .jl files that can be
`include()`d directly without NPZ dependency.

Output: superlam/assets/*.jl
"""

using NPZ

function write_jl_array(path::String, arr::AbstractArray, name::String)
    println("Writing $name to $path ($(size(arr)) $(eltype(arr)))")
    open(path, "w") do io
        println(io, "# Auto-generated - do not edit")
        println(io, "# Shape: $(size(arr)), Type: $(eltype(arr))")
        
        if ndims(arr) == 1
            println(io, "const $name = [")
            for (i, v) in enumerate(arr)
                print(io, "    ", v, i < length(arr) ? ",\n" : "\n")
            end
            println(io, "]")
        elseif ndims(arr) == 2
            # Julia matrix syntax: space = column, semicolon = row
            println(io, "const $name = [")
            for i in 1:size(arr, 1)
                row_vals = [repr(arr[i, j]) for j in 1:size(arr, 2)]
                row_str = join(row_vals, " ")
                print(io, "    ", row_str)
                print(io, i < size(arr, 1) ? ";\n" : ";\n")
            end
            println(io, "]")
        end
    end
end

function main()
    assets_dir = joinpath(@__DIR__, "assets")
    
    println("Reading .npy files from: ", assets_dir)
    
    R = npzread(joinpath(assets_dir, "R.npy"))
    neighbors = npzread(joinpath(assets_dir, "neighbors.npy"))
    A_em32 = npzread(joinpath(assets_dir, "A_em32.npy"))
    A_mic = npzread(joinpath(assets_dir, "A_mic.npy"))
    H_em32 = npzread(joinpath(assets_dir, "H_em32.npy"))
    H_mic = npzread(joinpath(assets_dir, "H_mic.npy"))
    
    println("\nWriting Julia source files...")
    write_jl_array(joinpath(assets_dir, "R.jl"), R, "R")
    write_jl_array(joinpath(assets_dir, "neighbors.jl"), neighbors, "neighbors")
    write_jl_array(joinpath(assets_dir, "A_em32.jl"), A_em32, "A_em32")
    write_jl_array(joinpath(assets_dir, "A_mic.jl"), A_mic, "A_mic")
    write_jl_array(joinpath(assets_dir, "H_em32.jl"), H_em32, "H_em32")
    write_jl_array(joinpath(assets_dir, "H_mic.jl"), H_mic, "H_mic")
    
    # Write a loader that includes all of them
    loader_path = joinpath(assets_dir, "assets_loader.jl")
    open(loader_path, "w") do io
        println(io, "# Auto-generated loader - includes all asset arrays")
        println(io, "include(\"R.jl\")")
        println(io, "include(\"neighbors.jl\")")
        println(io, "include(\"A_em32.jl\")")
        println(io, "include(\"A_mic.jl\")")
        println(io, "include(\"H_em32.jl\")")
        println(io, "include(\"H_mic.jl\")")
    end
    
    println("\nDone. Use: include(\"assets/assets_loader.jl\")")
end

main()
