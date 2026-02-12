abstract type AbstractParamNormalization{T} <: AbstractNormalization{T} end
const ParamNormUnion = Union{<:AbstractParamNormalization, Type{<:AbstractParamNormalization}}

macro _ParamNormalization(name, 𝑝, 𝑠, 𝑓)
    quote
        mutable struct $(esc(name)){T} <: AbstractParamNormalization{T}
            dims
            p::NTuple{length($(esc(𝑝))), AbstractArray{T}}
            s::NTuple{length($(esc(𝑠))), Real}
        end
        Normalization.estimators(::Type{N}) where {N<:$(esc(name))} = $(esc(𝑝))
        DataTreatments.parameters(::Type{N}) where {N<:$(esc(name))} = $(esc(𝑠))
        Normalization.forward(::Type{N}) where {N<:$(esc(name))} = $(esc(𝑓))
    end
end
parameters(::N) where {N<:AbstractParamNormalization} = parameters(N)
params(N::AbstractParamNormalization) = N.p
options(N::AbstractParamNormalization) = N.s

function __parammapdims!(z, f, x, y, o)
    @inbounds map!(f(map(only, y)..., o...), z, x)
end
function _parammapdims!(zs::Slices{<:AbstractArray}, f, xs::Slices{<:AbstractArray}, ys::NTuple{N, <:AbstractArray}, o) where {N}
    @sync Threads.@threads for i in eachindex(xs) #
        y = ntuple((j -> @inbounds ys[j][i]), Val(N)) # Extract parameters for nth slice
        __parammapdims!(zs[i], f, xs[i], y, o)
    end
end

function parammapdims!(z, f, x::AbstractArray{T, n}, y, o; dims) where {T, n}
    isnothing(dims) && (dims = 1:n)
     max(dims...) <= n || error("A chosen dimension is greater than the number of dimensions of the reference array")
    unique(dims) == [dims...] || error("Repeated dimensions")
    length(dims) == n && return __mapdims!(z, f, x, y) # ? Shortcut for global normalisation
    all(all(size.(y, i) .== 1) for i ∈ dims) || error("Inconsistent dimensions; dimensions $dims must have size 1")

    negs = negdims(dims, n)
    all(all(size(x, i) .== size.(y, i)) for i ∈ negs) || error("Inconsistent dimensions; dimensions $negs must have size $(size(x)[collect(negs)])")

    xs = eachslice(x; dims=negs)
    zs = eachslice(z; dims=negs)
    ys = eachslice.(y; dims=negs)
    _parammapdims!(zs, f, xs, ys, o)
end

# ---------------------------------------------------------------------------- #
#                                     fit                                      #
# ---------------------------------------------------------------------------- #
function fit!(T::AbstractParamNormalization, X::AbstractArray{A}; dims=Normalization.dims(T), kwargs...) where {A}
    eltype(T) == A || throw(TypeError(:fit!, "Normalization", eltype(T), X))

    dims, nps = dimparams(dims, X)
    Xs = eachslice(X; dims=negdims(dims, ndims(X)), drop=false)
    ps = map(estimators(T)) do f
        reshape(map(f, Xs), nps...)
    end

    dims!(T, dims)
    params!(T, ps)
    optparams!(T; kwargs...)
    nothing
end

function fit(::Type{𝒯}, X::AbstractArray{A}; dims=nothing, kwargs...) where {A,T,𝒯<:AbstractParamNormalization{T}}
    dims, nps = dimparams(dims, X)
    Xs = eachslice(X; dims=negdims(dims, ndims(X)), drop=false)
    ps = map(estimators(𝒯)) do f
        reshape(map(f, Xs), nps...)
    end
    𝒯(dims, ps; kwargs...)
end
function fit(::Type{𝒯}, X::AbstractArray{A}; kwargs...) where {A,𝒯<:AbstractParamNormalization}
    fit(𝒯{A}, X; kwargs...)
end
fit(N::AbstractParamNormalization, X::AbstractArray{A}; dims=Normalization.dims(N), kwargs...) where {A} = fit(typeof(N), X; dims, kwargs...)

# ---------------------------------------------------------------------------- #
#                                  normalize                                   #
# ---------------------------------------------------------------------------- #
function normalize!(Z::AbstractArray, X::AbstractArray, T::AbstractParamNormalization)
    dims = Normalization.dims(T)
    isfit(T) || fit!(T, X; dims)
    parammapdims!(Z, forward(T), X, params(T), options(T); dims)
    return nothing
end
function normalize!(Z, X, ::Type{𝒯}; kwargs...) where {𝒯 <: AbstractParamNormalization}
    normalize!(Z, X, fit(𝒯, X; kwargs...))
end
normalize!(X, T::ParamNormUnion; kwargs...) = normalize!(X, X, T; kwargs...)

function normalize(X, T::AbstractParamNormalization; kwargs...)
    Y = copy(X)
    normalize!(Y, T; kwargs...)
    return Y
end
function normalize(X, ::Type{𝒯}; kwargs...) where {𝒯 <: AbstractParamNormalization}
    normalize(X, fit(𝒯, X; kwargs...))
end

# ---------------------------------------------------------------------------- #
#                                    ZScore                                    #
# ---------------------------------------------------------------------------- #
@_Normalization ZScoreRobust (median, (x)->median(abs.(x .- median(x)))) zscore

function(::Type{ZScore})(method::Symbol)
    return if method == :robust
        ZScoreRobust
    elseif method == :half
        HalfZScore
    else
        ZScore
    end
end

# ---------------------------------------------------------------------------- #
#                                     Scale                                    #
# ---------------------------------------------------------------------------- #
@_Normalization Scale (std,) scale
@_Normalization ScaleMad ((x)->mad(x; normalize=false),) scale
@_Normalization ScaleFirst (first,) scale
@_Normalization ScaleIqr (iqr,) scale

scale(s) = Base.Fix2(/, s)

function(::Type{Scale})(factor::Symbol=:std)
    return if factor == :mad
        ScaleMad
    elseif factor == :first
        ScaleFirst
    elseif factor == :iqr
        ScaleIqr
    else
        Scale
    end
end

# ---------------------------------------------------------------------------- #
#                                    MinMax                                    #
# ---------------------------------------------------------------------------- #
@_ParamNormalization ScaledMinMax (minimum, maximum) (:lower, :upper) scaled_minmax

function optparams!(N::ScaledMinMax; lower::Real=0.0, upper::Real=1.0)
    normalization(N).s = (lower, upper)
end

(::Type{N})(
    dims=nothing,
    p=ntuple(_->Vector{T}(), length(estimators(N)));
    lower::Real=0.0,
    upper::Real=1.0
) where {T, N<:ScaledMinMax{T}} = N(dims, p, (lower, upper));

function scaled_minmax(xmin, xmax, lower, upper)
    scale = (upper - lower) / (xmax - xmin)
    (x) -> clamp(lower + (x - xmin) * scale, lower, upper)
end

# ---------------------------------------------------------------------------- #
#                                     Center                                   #
# ---------------------------------------------------------------------------- #
@_Normalization CenterMedian (median,) center

function(::Type{Center})(method::Symbol=:mean)
    return if method == :median
        CenterMedian
    else
        Center
    end
end

# ---------------------------------------------------------------------------- #
#                                     PNorm                                    #
# ---------------------------------------------------------------------------- #
@_Normalization PNorm ((x)->(norm(x, 1)),) scale
@_Normalization PNorm2 ((x)->(norm(x, 2)),) scale
@_Normalization PNormMax ((x)->(norm(x, Inf)),) scale

function(::Type{PNorm})(type::Symbol)
    return if type == :_1
        PNorm2
    elseif type == :max
        PNormMax
    else
        PNorm2
    end
end
