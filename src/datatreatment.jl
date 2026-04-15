# ---------------------------------------------------------------------------- #
#                          getter methods collection                           #
# ---------------------------------------------------------------------------- #
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
    any(ismissing.(data)) || (data = disallowmissing(data))

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
    perc::Real;
    include_nans::Bool=true,
    dims::Int=2
) where T
    @assert 0.0 ≤ perc ≤ 1.0 "perc must be between 0.0 and 1.0, got $perc"

    if dims == 1  # row-wise: global keep mask across ALL sub-datasets
        n = nrows(dt)
        total_cols = sum(ncols(d) for d in dt.data)
        row_badcount = zeros(Int, n)

        for d in dt.data
            missings = get_missingidxs.(d.info)
            missings = include_nans && !isa(d, DiscreteDataset) ?
                union.(missings, get_nanidxs.(d.info)) :
                missings
            foreach(idxs -> (row_badcount[idxs] .+= 1), missings)
        end

        keep = findall((row_badcount ./ total_cols) .≤ perc)

        data = map(dt.data) do d
            new_info = _reindex_feat.(d.info, Ref(keep))

            isa(d, MultidimDataset{<:Any, AggregateFeat}) ?
                typeof(d).name.wrapper(d.data[keep, :], new_info, d.groups) :
            isa(d, MultidimDataset{<:Any, ReduceFeat}) ?
                typeof(d).name.wrapper(d.data[keep, :], new_info, d.groups) :
                typeof(d).name.wrapper(d.data[keep, :], new_info)
        end

        target = get_target(dt)
        target = isempty(target) ? target : target[keep]

        return DataTreatment{T}(data, target, get_treats(dt), get_balance(dt))

    else  # col-wise: independent per sub-dataset (no row consistency issue)
        data = map(dt.data) do d
            missings = get_missingidxs.(d.info)
            missings = include_nans && !isa(d, DiscreteDataset) ?
                union.(missings, get_nanidxs.(d.info)) :
                missings

            n = nrows(d)
            keep = (length.(missings) ./ n) .≤ perc

            isa(d, MultidimDataset{<:Any, AggregateFeat}) ?
                typeof(d).name.wrapper(d.data[:, keep], d.info[keep], d.groups) :
                typeof(d).name.wrapper(d.data[:, keep], d.info[keep])
        end

        return DataTreatment{T}(
            data, get_target(dt), get_treats(dt), get_balance(dt))
    end
end

