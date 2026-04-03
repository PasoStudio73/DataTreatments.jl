# ---------------------------------------------------------------------------- #
#                             DataTreatment struct                             #
# ---------------------------------------------------------------------------- #
"""
    DataTreatment{T}

Top-level container produced by [`load_dataset`](@ref). Holds all output datasets
derived from a source table, a target vector, and the applied treatment groups.

## Structure

```
DataTreatment{T}  (T = float_type, e.g. Float64)
├── data    ::Vector{AbstractDataset}   # ordered list of output datasets
│    ├── DiscreteDataset{...}           # from discrete (categorical) columns
│    ├── ContinuousDataset{T}           # from continuous (scalar) columns
│    └── MultidimDataset{T, ...}        # from multidimensional columns
│         ├── AggregateFeat variant     # → tabular scalar output
│         └── ReduceFeat variant        # → array output
├── target  ::AbstractVector            # encoded target vector (labels)
└── treats  ::Vector{TreatmentGroup}    # user directives
```

## Full pipeline overview

```
Raw DataFrame / Matrix
        │
        ▼  load_dataset(data, vnames, target, treatments...; float_type=Float64)
        │
        ├─ _inspecting(data)  →  datastruct::NamedTuple
        │   (inspect all columns: types, missing, NaN, dims, ...)
        │
        ├─ encode target  →  CategoricalVector
        │
        ├─ for each TreatmentGroup:
        │   ├── classify columns  →  discrete_ids, continuous_ids, multidim_ids
        │   ├── DiscreteDataset(discrete_ids, ...)
        │   ├── ContinuousDataset(continuous_ids, ...)
        │   └── MultidimDataset(multidim_ids, ..., aggrfunc)
        │
        └── DataTreatment{T}(ds, target, treats)
```

## Accessor summary

| Method              | Returns                                      |
|---------------------|----------------------------------------------|
| `get_discrete(dt)`  | `(Matrix, vnames)` for categorical columns   |
| `get_continuous(dt)`| `(Matrix{T}, vnames)` for scalar columns     |
| `get_aggregated(dt)`| `(Matrix{T}, vnames)` for aggregated series  |
| `get_reduced(dt)`   | `(Matrix{Array{T}}, vnames)` for series      |
| `get_tabular(dt)`   | merged tabular matrix + all column names     |
| `get_multidim(dt)`  | reduced multidim matrix + column names       |
| `get_target(dt)`    | the encoded target vector                    |

# Type Parameter
- `T`: The floating-point type used throughout (e.g., `Float64`, `Float32`).

See also: [`load_dataset`](@ref), [`DiscreteDataset`](@ref),
[`ContinuousDataset`](@ref), [`MultidimDataset`](@ref), [`TreatmentGroup`](@ref)
"""
mutable struct DataTreatment{T}
    data::Vector{AbstractDataset}
    target::AbstractVector
    treats::Vector{TreatmentGroup}
end

nrows(dt::DataTreatment) = size(first(dt.data).data, 1)

get_target(dt::DataTreatment) = dt.target
get_treats(dt::DataTreatment) = dt.treats

function get_discrete(
    dt::DataTreatment
)
    ds = collect(filter(d -> d isa DiscreteDataset, dt.data))
    return if isempty(ds)
        Matrix{Union{Missing,CategoricalValue}}(undef, 0, 0), String[]
    else
        get_data(ds), reduce(vcat, get_vnames.(ds))
    end
end

function get_continuous(
    dt::DataTreatment{T}
) where {T<:Float}
    ds = collect(filter(d -> d isa ContinuousDataset{T}, dt.data))
    return if isempty(ds)
        Matrix{T}(undef, 0, 0), String[]
    else
        get_data(ds), reduce(vcat, get_vnames.(ds))
    end
end

function get_aggregated(
    dt::DataTreatment{T}
) where {T<:Float}
    ds = filter(d -> d isa MultidimDataset &&
        all(elt -> elt isa AggregateFeat{T}, get_info(d)), dt.data)
    return if isempty(ds)
        Matrix{T}(undef, 0, 0), String[]
    else
        (get_data(ds)), reduce(vcat, get_vnames.(ds))
    end
end

function get_reduced(
    dt::DataTreatment{T}
) where {T<:Float}
    ds = filter(d -> d isa MultidimDataset &&
        all(elt -> elt isa ReduceFeat, get_info(d)), dt.data)
    return if isempty(ds)
        (Matrix{VecOrMat{T}}(undef, 0, 0), String[])
    else
        (get_data(ds)), reduce(vcat, get_vnames.(ds))
    end
end

is_tabular(dt::DataTreatment) = all(is_tabular.(dt.data))
is_multidim(dt::DataTreatment) = all(is_multidim.(dt.data))

has_tabular(dt::DataTreatment) = any(is_tabular.(dt.data))
has_multidim(dt::DataTreatment) = any(is_multidim.(dt.data))

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
function load_dataset(
    data::AbstractMatrix{T},
    vnames::Vector{String}=["V$i" for i in 1:size(data, 2)],
    target::Union{Nothing,AbstractVector}=nothing,
    treatments::Vararg{Base.Callable}=DefaultTreatmentGroup;
    treatment_ds::Bool=true,
    leftover_ds::Bool=false,
    float_type::Type=Float64
) where T
    datastruct = _inspecting(data)

    if isnothing(target)
        target = CategoricalVector[]
    elseif !isnothing(target) && !(eltype(target) <: AbstractFloat)
        target = _discrete_encode(target)
    end

    treats = [treat(datastruct, vnames) for treat in treatments]

    ds = AbstractDataset[]

    if treatment_ds
        ds_td, ds_tc, ds_md = _treatments_ds(
            data,
            vnames,
            datastruct,
            treats, 
            float_type
        )
        !isempty(ds_td) && append!(ds, ds_td)
        !isempty(ds_tc) && append!(ds, ds_tc)
        !isempty(ds_md) && append!(ds, ds_md)
    end

    if leftover_ds
        ds_td, ds_tc, ds_md = _leftovers_ds(
            data,
            vnames,
            datastruct,
            treats,
            float_type
        )
        !isempty(ds_td) && append!(ds, ds_td)
        !isempty(ds_tc) && append!(ds, ds_tc)
        !isempty(ds_md) && append!(ds, ds_md)
    end

    return DataTreatment{float_type}(ds, target, treats)
end

load_dataset(df::DataFrame, target::AbstractVector, args...; kwargs...) =
    load_dataset(Matrix(df), names(df), target, args...; kwargs...)

load_dataset(df::DataFrame, args...; kwargs...) =
    load_dataset(Matrix(df), names(df), CategoricalVector[], args...; kwargs...)

function load_dataset(
    dt::DataTreatment{T},
    treatments::Vararg{Base.Callable}=DefaultTreatmentGroup;
    kwargs...
) where T
    data = reduce(hcat, d.data for d in dt.data)
    vnames = reduce(vcat, get_vnames(d) for d in dt.data)
    target = get_target(dt)

    load_dataset(data, vnames, target, treatments...; kwargs...)
end

# ---------------------------------------------------------------------------- #
#                             get tabular method                               #
# ---------------------------------------------------------------------------- #
"""
    get_tabular(dt::DataTreatment)

Convenience function to collect all tabular-like datasets from a `DataTreatment` 
object, including discrete, continuous, and aggregated multidimensional data.
"""
@inline function get_tabular(
    dt::DataTreatment{T}
) where {T<:Float}
    mats = get_discrete(dt), get_continuous(dt), get_aggregated(dt)
    idxs = findall(x -> !(isempty(x)), map(first, mats))

    isempty(idxs) && return(
        (Matrix{T}(undef, 0,0), String[])
    )

    X = collect(zip(mats[idxs]...))
    Tnew = unique(eltype.(X[1]))
    data = Matrix{Union{Tnew...}}(reduce(hcat, X[1]))

    return (data, reduce(vcat, X[2]))
end

# ---------------------------------------------------------------------------- #
#                            get multidim method                               #
# ---------------------------------------------------------------------------- #
"""
    get_multidim(dt::DataTreatment)

Convenience function to collect all reduced multidimensional datasets 
from a `DataTreatment` object.
"""
@inline function get_multidim(
    dt::DataTreatment{T};
    kwargs...
) where {T<:Float}
    data, vnames = get_reduced(dt; kwargs...)
    any(ismissing.(data)) || (data = disallowmissing(data))

    return data, vnames
end

# ---------------------------------------------------------------------------- #
#                         filter missing by percentage                         #
# ---------------------------------------------------------------------------- #
function filter_missing(
    dt::DataTreatment{T},
    perc::AbstractFloat;
    include_nans::Bool=true,
    dims::Int=2
) where T
    @assert 0.0 ≤ perc ≤ 1.0 "perc must be between 0.0 and 1.0, got $perc"

    data = map(dt.data) do d
        missings = get_missingidxs.(d.info)
        missings = include_nans && !isa(d, DiscreteDataset) ?
            union.(missings, get_nanidxs.(d.info)) :
            missings

        n = nrows(d)

        if dims == 1  # row-wise
            nc = ncols(d)
            row_badcount = zeros(Int, n)
            foreach(idxs -> (row_badcount[idxs] .+= 1), missings)
            keep = findall((row_badcount ./ nc) .≤ perc)
            new_info = _reindex_feat.(d.info, Ref(keep))

            isa(d, MultidimDataset{<:Any, AggregateFeat}) ?
                typeof(d).name.wrapper(d.data[keep, :], new_info, d.groups) :
            isa(d, MultidimDataset{<:Any, ReduceFeat}) ?
                typeof(d).name.wrapper(d.data[keep, :], new_info, d.groups) :
                typeof(d).name.wrapper(d.data[keep, :], new_info)
        else          # col-wise
            keep = (length.(missings) ./ n) .≤ perc
            isa(d, MultidimDataset{<:Any, AggregateFeat}) ?
                typeof(d).name.wrapper(d.data[:, keep], d.info[keep], d.groups) :
                typeof(d).name.wrapper(d.data[:, keep], d.info[keep])
        end
    end

    return DataTreatment{T}(data, get_target(dt), get_treats(dt))
end

