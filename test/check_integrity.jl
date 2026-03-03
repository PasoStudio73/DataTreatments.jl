using Test
using DataTreatments
const DT = DataTreatments

using DataFrames
using CategoricalArrays
using Random

function create_image(seed::Int)
    Random.seed!(seed)
    rand(Float64, 6, 6)
end

df = DataFrame(
    str_col  = [missing, "blue", "green", "red", "blue"],
    sym_col  = [:circle, :square, :triangle, :square, missing],
    cat_col  = categorical(["small", "medium", missing, "small", "large"]),
    uint_col = UInt32[1, 2, 3, 4, 5],
    int_col  = Int[10, 20, 30, 40, 50],
    V1 = [missing, 2.0, 3.0, 4.0, 5.6],
    V2 = [2.5, 3.5, 4.5, 5.5, missing],
    V3 = [3.2, 4.2, 5.2, 6.2, 2.4],
    V4 = [4.1, missing, missing, 7.1, 5.5],
    V5 = [5.0, 6.0, 7.0, 8.0, 1.8],
    ts1 = [missing, collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
    ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), missing],
    ts3 = [collect(1.0:1.2:7.0), missing, missing, collect(1.5:1.2:7.5), collect(3.0:1.2:9.0)],
    ts4 = [collect(6.0:-0.8:1.0), collect(7.0:-0.8:2.0), collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), collect(9.0:-0.8:4.0)],
    img1 = [create_image(i) for i in 1:5],
    img2 = [i == 1 ? missing : create_image(i+10) for i in 1:5],
    img3 = [create_image(i+20) for i in 1:5],
    img4 = [i == 3 ? missing : create_image(i+30) for i in 1:5]
)

X = Matrix(df)

function base_eltype(col::AbstractArray)
    valtype, idx, hasmissing, hasnan = nothing, nothing, false, false
    
    # First pass: scan entire column to find actual value type
    for i in eachindex(col)
        val = col[i]
        if ismissing(val)
            hasmissing = true
        elseif val isa AbstractFloat && isnan(val)
            hasnan = true
        elseif val isa AbstractVector{<:AbstractFloat} || val isa AbstractArray{<:AbstractFloat}
            if any(isnan, val)
                hasnan = true
            end
            valtype = typeof(val)
            idx = i
        elseif !(val isa AbstractFloat) || !isnan(val)  # Skip NaN scalars
            if isnothing(valtype) || !(valtype <: AbstractVector)
                valtype = typeof(val)
                idx = i
            end
        end
    end
    
    return valtype, idx, hasmissing, hasnan
end

function check_integrity(X::Matrix)
    dim = size(X, 2)
    valtype = Vector{Type}(undef, dim)
    idx = Vector{Int}(undef, dim)
    hasmissing = Vector{Bool}(undef, dim)
    hasnan = Vector{Bool}(undef, dim)

    Threads.@threads for i in axes(X, 2)
        valtype[i], idx[i], hasmissing[i], hasnan[i] = base_eltype(@view(X[:, i]))
    end
    return valtype, idx, hasmissing, hasnan
end