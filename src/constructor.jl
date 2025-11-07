# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
# base type for metadata containers
abstract type AbstractDataTreatment end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const ValidVnames = Union{Symbol, String}

# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
# recursively extract the core element type from nested array types.
core_eltype(x) = eltype(x) <: AbstractArray ? core_eltype(eltype(x)) : eltype(x)

is_multidim_dataframe(X::AbstractArray)::Bool =
    any(eltype(col) <: AbstractArray for col in eachcol(X))

# ---------------------------------------------------------------------------- #
#                                 constructor                                  #
# ---------------------------------------------------------------------------- #
struct DataTreatment <: AbstractDataTreatment
    dataset    :: AbstractMatrix
    vnames     :: Vector{Symbol}
    intervals  :: Tuple{Vararg{Vector{UnitRange{Int64}}}}
    features   :: Tuple{Vararg{Base.Callable}}
    reducefunc :: Base.Callable
    aggrtype   :: Symbol

    function DataTreatment(
        X          :: AbstractMatrix,
        aggrtype   :: Symbol;
        vnames     :: Vector{<:ValidVnames},
        win        :: Union{Base.Callable, Tuple{Vararg{Base.Callable}}},
        features   :: Tuple{Vararg{Base.Callable}}=(maximum, minimum, mean),
        reducefunc :: Base.Callable=mean,
        norm       :: Bool=false
    )
        is_multidim_dataframe(X) || throw(ArgumentError("Input DataFrame " * 
            "does not contain multidimensional data."))

        vnames isa Vector{String} && (vnames = Symbol.(vnames))
        win isa Base.Callable && (win = (win,))

        vnames isa Vector{String} && (vnames = Symbol.(vnames))
        intervals = @evalwindow first(X) win...
        nwindows = prod(length.(intervals))

        Xresult, colnames = if aggrtype == :aggregate begin
            (aggregate(X, intervals; features),
            if nwindows == 1
                # single window: apply to whole time series
                [Symbol("$(f)($(v))") for f in features, v in vnames] |> vec
            else
                # multiple windows: apply to each interval
                [Symbol("$(f)($(v))_w$(i)") 
                for i in 1:nwindows, f in features, v in vnames] |> vec
            end
            )
        end

        elseif aggrtype == :reducesize begin
            (reducesize(X, intervals; reducefunc),
            vnames
            )
        end

        else
            error("Unknown treatment type: $treat")
        end

        new(Xresult, colnames, intervals, features, reducefunc, aggrtype)
    end

    function DataTreatment(
        X      :: AbstractDataFrame,
        args...;
        vnames :: Union{Vector{<:ValidVnames}, Nothing}=nothing,
        kwargs...
    )
        isnothing(vnames) && (vnames = propertynames(X))
        DataTreatment(Matrix(X), args...; vnames, kwargs...)
    end
end
