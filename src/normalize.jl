# ---------------------------------------------------------------------------- #
#                                @_Normalization                               #
# ---------------------------------------------------------------------------- #
@_Normalization ZScoreRobust (median, (x)->median(abs.(x .- median(x)))) Normalization.zscore

scale(s) = Base.Fix2(/, s)
@_Normalization Scale (std,) scale
@_Normalization ScaleMad ((x)->mad(x; normalize=false),) scale
@_Normalization ScaleFirst (first,) scale
@_Normalization ScaleIqr (iqr,) scale

@_Normalization CenterMedian (median,) Normalization.center

@_Normalization PNorm1 ((x)->norm(x, 1),) scale
@_Normalization PNorm ((x)->norm(x, 2),) scale
@_Normalization PNormInf ((x)->norm(x, Inf),) scale

normalize(X, p::NamedTuple) = normalize(X, p[1]; Base.tail(p)...)

# ---------------------------------------------------------------------------- #
#                                    callers                                   #
# ---------------------------------------------------------------------------- #
checkdims(dims::Union{Int64,Nothing}=nothing) =
    (isnothing(dims) || dims == 1 || dims == 2) ||
        error("dims must be nothing, 1 (column-wise), or 2 (row-wise)")

checkmethod(method::Symbol, methods::Tuple{Vararg{Symbol}}) =
    (method in methods) || error("method must be $methods")

checkp(p::Real) =
    (p == 1 || p == 2 || p == Inf) || error("p must be 1, 2, or Inf")

function (::Type{ZScore})(; dims::Union{Int64,Nothing}=nothing, method::Symbol=:std)
    checkdims(dims)
    checkmethod(method, (:std, :robust, :half))
    method == :robust && return (type = ZScoreRobust, dims = dims)
    method == :half && return (type = HalfZScore, dims = dims)
    return (type = ZScore, dims = dims)
end

function (::Type{MinMax})(; dims::Union{Int64,Nothing}=nothing)
    checkdims(dims)
    return (type = MinMax, dims = dims)
end

function (::Type{Scale})(; dims::Union{Int64,Nothing}=nothing, method::Symbol=:std)
    checkdims(dims)
    checkmethod(method, (:std, :mad, :first, :iqr))
    method == :mad && return (type = ScaleMad, dims = dims)
    method == :first && return (type = ScaleFirst, dims = dims)
    method == :iqr && return (type = ScaleIqr, dims = dims)
    return (type = Scale, dims = dims)
end

function (::Type{Sigmoid})(; dims::Union{Int64,Nothing}=nothing)
    checkdims(dims)
    return (type = Sigmoid, dims = dims)
end

function (::Type{Center})(; dims::Union{Int64,Nothing}=nothing, method::Symbol=:mean)
    checkdims(dims)
    checkmethod(method, (:mean, :median))
    method == :median && return (type = CenterMedian, dims = dims)
    return (type = Center, dims = dims)
end

function (::Type{UnitEnergy})(; dims::Union{Int64,Nothing}=nothing)
    checkdims(dims)
    return (type = UnitEnergy, dims = dims)
end

function (::Type{UnitPower})(; dims::Union{Int64,Nothing}=nothing)
    checkdims(dims)
    return (type = UnitPower, dims = dims)
end

function (::Type{PNorm})(; dims::Union{Int64,Nothing}=nothing, p::Real=2.0)
    checkdims(dims)
    checkp(p)
    p == 1 && return (type = PNorm1, dims = dims)
    p == Inf && return (type = PNormInf, dims = dims)
    return (type = PNorm, dims = dims)
end
