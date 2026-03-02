using Test
using DataTreatments

using DataFrames
using CategoricalArrays

# ---------------------------------------------------------------------------- #
#                     dataset with only discrete features                      #
# ---------------------------------------------------------------------------- #
df = DataFrame(
    str_col  = ["red", "blue", "green", "red", "blue"],                        # AbstractString
    sym_col  = [:circle, :square, :triangle, :square, :circle],                # Symbol
    cat_col  = categorical(["small", "medium", "large", "small", "large"]),    # CategoricalValue
    uint_col = UInt32[1, 2, 3, 4, 5],                                          # UInt32
    int_col  = Int[10, 20, 30, 40, 50]                                         # Int
)

dt = DataTreatment(df)

df = DataFrame(
    str_col  = [missing, missing, "green", "red", "blue"],                     # AbstractString
    sym_col  = [missing, :square, :triangle, :square, :circle],                # Symbol
    cat_col  = categorical([missing, "medium", "large", "small", missing]),    # CategoricalValue
    uint_col = Union{Missing, UInt32}[missing, 2, 3, 4, 5],                    # UInt32
    int_col  = Union{Missing, Int}[missing, 20, 30, 40, 50]                    # Int
)

dt = DataTreatment(df)


X = Matrix(df)

@btime codes = [levelcode.(categorical(string.(col))) for col in eachcol(X)]
# 10.564 μs (208 allocations: 10.69 KiB)

@btime begin
codes, lvls = map(eachcol(X)) do col
    cat = categorical(string.(col))
    levelcode.(cat), levels(cat)
end
end
# 10.788 μs (225 allocations: 11.27 KiB)

function discrete_encode(X::Matrix)
    to_str(v) = (ismissing(v) || (v isa AbstractFloat && isnan(v))) ? missing : string(v)
    cats = [categorical(to_str.(col)) for col in eachcol(X)]
    return [levelcode.(cat) for cat in cats], levels.(cats)
end

@btime begin
    cats = map(eachcol(X)) do col
        categorical(map(v -> ismissing(v) ? missing : string(v), col))
    end
end

    @btime begin
cats = [categorical(map(v -> (ismissing(v) || (v isa AbstractFloat && isnan(v))) ? missing : string(v), col)) for col in eachcol(X)]
    end

@btime begin
to_str(v) = (ismissing(v) || (v isa AbstractFloat && isnan(v))) ? missing : string(v)
cats = [categorical(to_str.(col)) for col in eachcol(X)]
end