# ---------------------------------------------------------------------------- #
#                              split & _groupby                                #
# ---------------------------------------------------------------------------- #
@inline field_getter(field::Symbol) =
    field == :dims ? f -> f.dims :
    field == :vname ? f -> f.vname :
    field == :nwin ? f -> f.nwin :
    field == :feat ? f -> f.feat :
    throw(ArgumentError("Unknown field: $field"))

function _groupby(
    info::AbstractVector{<:AggregateFeat{T}},
    fields::Tuple{Vararg{Symbol}}
) where T
    sub_idxs = _groupby(info, first(fields))

    remaining = fields[2:end]
    isempty(remaining) && return collect(sub_idxs)

    all_groups = Vector{Vector{Int}}()

    for i in sub_idxs
        groups = _groupby(@view(info[i]), remaining)
        append!(all_groups, collect(groups))
    end

    return all_groups
end

function _groupby(
    info::AbstractVector{<:AggregateFeat{T}},
    field::Symbol
) where T
    field == :all && return (i for i in eachindex(info))
    getter = field_getter(field)
    vals = [getter(info[i]) for i in eachindex(info)]
    unique_vals = unique(vals)
    idxs = (findall(==(v), vals) for v in unique_vals)
    return (get_subid.(info[i]) for i in idxs)
end
