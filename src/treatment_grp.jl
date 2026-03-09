using DataFrames
using Random
using CategoricalArrays

function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

df = DataFrame(
    str_col  = [missing, "blue", "green", "red", "blue"],
    sym_col  = [:circle, :square, :triangle, :square, missing],
    cat_col  = categorical(["small", "medium", missing, "small", "large"]),
    uint_col = UInt32[1, 2, 3, 4, 5],
    int_col  = Int[10, 20, 30, 40, 50],
    V1 = [NaN, missing, 3.0, 4.0, 5.6],
    V2 = [2.5, missing, 4.5, 5.5, NaN],
    V3 = [3.2, 4.2, 5.2, missing, 2.4],
    V4 = [4.1, NaN, NaN, 7.1, 5.5],
    V5 = [5.0, 6.0, 7.0, 8.0, 1.8],
    ts1 = [NaN, collect(2.0:7.0), missing, collect(4.0:9.0), collect(5.0:10.0)],
    ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), NaN],
    ts3 = [collect(1.0:1.2:7.0), NaN, NaN, missing, collect(3.0:1.2:9.0)],
    ts4 = [collect(6.0:-0.8:1.0), missing, collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), collect(9.0:-0.8:4.0)],
    img1 = [create_image(i) for i in 1:5],
    img2 = [i == 1 ? NaN : create_image(i+10) for i in 1:5],
    img3 = [create_image(i+20) for i in 1:5],
    img4 = [i == 3 ? missing : create_image(i+30) for i in 1:5]
)

abstract type AbstractDataFeature end

include("../src/errors.jl")
using DataTreatments: reducesize, wholewindow, @evalwindow
include("../src/structs/dataset_structure.jl")

function discrete_encode(X::Matrix)
    to_str(v) = (ismissing(v) || (v isa AbstractFloat && isnan(v))) ? missing : string(v)
    cats = [categorical(to_str.(col)) for col in eachcol(X)]
    return [levelcode.(cat) for cat in cats], levels.(cats)
end

# ---------------------------------------------------------------------------- #
#                                DataFeature                                   #
# ---------------------------------------------------------------------------- #
struct DiscreteFeat{T} <: AbstractDataFeature
    id::Int
    vname::Symbol
    levels::CategoricalArrays.CategoricalVector
    valididxs::Vector{Int}
    missingidxs::Vector{Int}

    function DiscreteFeat{T}(
        id::Int,
        vname::Union{String,Symbol},
        levels::CategoricalArrays.CategoricalVector,
        valididxs::Vector{Int},
        missingidxs::Vector{Int}
    ) where T
        new{T}(id, Symbol(vname), levels, valididxs, missingidxs)
    end
end

struct ContinuousFeat{T} <: AbstractDataFeature
    id::Int
    vname::Symbol
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}

    function ContinuousFeat{T}(
        id::Int,
        vname::Union{String,Symbol},
        valididxs::Vector{Int},
        missingidxs::Vector{Int},
        nanidxs::Vector{Int}
    ) where T
        new{T}(id, Symbol(vname), valididxs, missingidxs, nanidxs)
    end
end

struct AggregateFeat{T} <: AbstractDataFeature
    id::Int
    vname::Symbol
    feat::Base.Callable
    nwin::Int
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}
    # hasmissing::Vector{Bool}}
    # hasnans::Vector{Bool}}
end

struct ReduceFeat{T} <: AbstractDataFeature
    id::Int
    vname::Symbol
    reducefunc::Base.Callable
    valididxs::Vector{Int}
    missingidxs::Vector{Int}
    nanidxs::Vector{Int}
    # hasmissing::Vector{Bool}}
    # hasnans::Vector{Bool}}
end

######################################################################

# function build_datasets(
#     dataset::Matrix,
#     ds_struct::DatasetStructure,
#     vnames::Vector{String},
#     float_type::Type
# )
#     dstd, dstc, dsmd = nothing, nothing, nothing
#     valtype = get_datatype(ds_struct)

#     td_cols = findall(T -> !isnothing(T) && !(T <: AbstractFloat) && !(T <: AbstractArray), valtype)
#     tc_cols = findall(T -> !isnothing(T) && T <: AbstractFloat, valtype)
#     md_cols = findall(T -> !isnothing(T) && T <: AbstractArray, valtype)

#     # discrete
#     if !isempty(td_cols)
#         vnames_td = @views vnames[td_cols]
#         types_td = get_datatype(ds_struct, td_cols)
#         valid_td = get_valididxs(ds_struct, td_cols)
#         miss_td = get_missingidxs(ds_struct, td_cols)
#         codes, levels = discrete_encode(dataset[:, td_cols])

#         dstd = stack(codes)
#         td_feats = [DiscreteFeat{types_td[i]}(i, vnames_td[i], levels[i], valid_td[i], miss_td[i])
#             for i in eachindex(vnames_td)]
#     end

#     # continue
#     if !isempty(tc_cols)
#         vnames_tc = @views vnames[tc_cols]
#         valid_td = get_valididxs(ds_struct, td_cols)
#         miss_tc = get_missingidxs(ds_struct, tc_cols)
#         nan_tc = get_nanidxs(ds_struct, tc_cols)

#         dstc = reduce(hcat, [map(x -> ismissing(x) ? missing : float_type(x), @view dataset[:, col])
#             for col in tc_cols])
#         tc_feats = [ContinuousFeat{float_type}(i, vnames_tc[i], valid_td[i], miss_tc[i], nan_tc[i])
#             for i in eachindex(vnames_tc)]
#     end

#     @show tc_feats

#     # multidimensional
#     if !isempty(md_cols)
#         X = @view X[:, md_cols]
#         vnames_md = @views vnames[md_cols]
#         idx_md = @views idx[md_cols]
#         miss, nan = hasmissing[md_cols], hasnan[md_cols]
#         win isa Base.Callable && (win = (win,))

#         if aggrfunc == :aggregate
#             dsmd, nwindows = DataTreatments.aggregate(X, win, features, idx_md, float_type)
#             md_feats = vec([AggregateFeat{float_type}(i, vnames_md[c], f, nwindows[c], miss[c], nan[c])
#                     for (i, (f, c)) in enumerate(Iterators.product(features, axes(X,2)))])

#         elseif aggrfunc == :reducesize
#             dsmd = DataTreatments.reducesize(X, win, reducefunc, idx_md, float_type)
#             md_feats = [ReduceFeat{AbstractArray{float_type}}(i, vnames_md[c], reducefunc, miss[c], nan[c])
#                 for (i, c) in enumerate(axes(X,2))]

#         else
#             error("Unknown treatment type: $treat")
#         end
#     end
# end
function aggregate(
    X::AbstractArray,
    idx::AbstractVector{Vector{Int}},
    float_type::Type;
    win::Tuple{Vararg{Base.Callable}},
    features::Tuple{Vararg{Base.Callable}},
)
    colwin = [[n > length(win) ? last(win) : win[n] for n in 1:ndims(X[first(idx[i]), i])] for i in axes(X, 2)]
    nwindows = [prod(hasfield(typeof(w), :nwindows) ? w.nwindows : 1 for w in c) for c in colwin]
    nfeats = length(features)
# vettore di indici
    Xa = Matrix{Union{Missing,float_type}}(undef, size(X, 1), sum(nwindows) * nfeats)
    outtmp = 1

    @inbounds for colidx in axes(X, 2)
        outidx = outtmp

        for rowidx in axes(X, 1)
            x = X[rowidx, colidx]
            outidx = outtmp

            if rowidx in idx[colidx]
                intervals = @evalwindow X[rowidx, colidx] colwin[colidx]...
                for feat in features
                    for cartidx in CartesianIndices(length.(intervals))
                        ranges = get_window_ranges(intervals, cartidx)
                        window_view = @views x[ranges...]
                        Xa[rowidx, outidx] = safe_feat(reshape(window_view, :), feat)
                        outidx += 1
                    end
                end
            else
                intervals = nwindows[colidx] * nfeats
                Xa[rowidx, outidx:outidx+intervals-1] .= ismissing(x) ? x : float_type(x)
                outidx += intervals
            end
        end

        outtmp = outidx
    end

    return Xa, nwindows
end

aggregate(; kwargs...) = (x, i, ft) -> aggregate(x, i, ft; kwargs...)

# ---------------------------------------------------------------------------- #
#                            TreatmentGroup struct                             #
# ---------------------------------------------------------------------------- #
struct TreatmentGroup{T}
    idxs::Vector{Int}
    dims::Int
    vnames::Vector{String}
    aggrfunc::Base.Callable
    groupby::Tuple{Vararg{Symbol}}

    function TreatmentGroup(
        ds_struct::DatasetStructure;
        dims::Int=-1,
        name_expr::Union{Regex,Base.Callable}=r".*",
        datatype::Type=Any,
        aggrfunc::Base.Callable=aggregate(win=(wholewindow(),), features=(maximum, minimum, mean)),
        groupby::Tuple{Vararg{Symbol}}=(:vname,)
    )
        # filter by dims
        idxs = dims == -1 ?
            collect(1:length(ds_struct)) :
            findall(get_dims(ds_struct) .== dims)

        # filter by names
        vnames = get_vnames(ds_struct)
        valid_names = name_expr isa Regex ?
            filter(item -> match(name_expr, item) !== nothing, vnames) :
            filter(name_expr, vnames)
        idxs = idxs ∩ findall(n -> n in valid_names, vnames)

        # filter by datatype
        datatype != Any && (idxs= idxs ∩ findall(get_datatype(ds_struct) .== datatype))

        new{datatype}(idxs, dims, valid_names, aggrfunc, groupby)
    end

    # TreatmentGroup(vnames::Vector{Symbol}; kwargs...) = TreatmentGroup(String.(vnames); kwargs...)

    # TreatmentGroup(df::DataFrame; kwargs...) = TreatmentGroup(get_dataset_structure(df); kwargs...)
end

TreatmentGroup(; kwargs...) = x -> TreatmentGroup(x; kwargs...)

# ---------------------------------------------------------------------------- #
#                                Base methods                                  #
# ---------------------------------------------------------------------------- #
"""
    Base.length(tg::TreatmentGroup)

Returns the number of columns selected by this group.
"""
Base.length(tg::TreatmentGroup) = length(tg.idxs)

"""
    Base.iterate(tg::TreatmentGroup, state=1)

Iterates over the selected column indices.
"""
Base.iterate(tg::TreatmentGroup, state=1) = state > length(tg) ? nothing : (tg.idxs[state], state + 1)

"""
    Base.eachindex(tg::TreatmentGroup)

Returns the indices of the selected columns vector.
"""
Base.eachindex(tg::TreatmentGroup) = eachindex(tg.idxs)

# ---------------------------------------------------------------------------- #
#                               getter methods                                 #
# ---------------------------------------------------------------------------- #
"""
    get_idxs(tg::TreatmentGroup)
    get_idxs(tg::TreatmentGroup, i::Int)

Returns the column indices selected by this group. If `i` is provided, returns the `i`-th index.
"""
get_idxs(tg::TreatmentGroup) = tg.idxs
get_idxs(tg::TreatmentGroup, i::Int) = tg.idxs[i]

"""
    get_dims(tg::TreatmentGroup)

Returns the dimensionality filter used to select columns (`-1` means no filter).
"""
get_dims(tg::TreatmentGroup) = tg.dims

"""
    get_vnames(tg::TreatmentGroup)
    get_vnames(tg::TreatmentGroup, i::Int)
    get_vnames(tg::TreatmentGroup, idxs::Vector{Int})

Returns the column names selected by this group. If `i` is provided, returns the `i`-th name.
If a vector of indices is provided, returns a view of the names for those positions.
"""
get_vnames(tg::TreatmentGroup) = tg.vnames
get_vnames(tg::TreatmentGroup, i::Int) = tg.vnames[i]
get_vnames(tg::TreatmentGroup, idxs::Vector{Int}) = @views tg.vnames[idxs]

"""
    get_aggrfunc(tg::TreatmentGroup)

Returns the aggregation function used by this group.
"""
get_aggrfunc(tg::TreatmentGroup) = tg.aggrfunc

"""
    get_groupby(tg::TreatmentGroup)

Returns the `groupby` tuple of symbols used to partition output features.
"""
get_groupby(tg::TreatmentGroup) = tg.groupby

######################################################################

function DataTreatment(
    dataset::Matrix,
    vnames::Vector{String},
    treatments::Base.Callable...=TreatmentGroup(
        aggrfunc=aggregate(win=(wholewindow(),), features=(maximum, minimum, mean)),
    );
    float_type::Type=Float64
)
    ds_struct = get_dataset_structure(dataset, vnames)
    tgroups = [treat(ds_struct) for treat in treatments]
    # build_datasets(dataset, ds_struct, vnames, float_type)

end

DataTreatment(df::DataFrame, args...; kwargs...) =
    DataTreatment(Matrix(df), names(df), args...; kwargs...)

########################################################################

# test = DataTreatment(df, TreatmentGroup(dims=1, name_expr=r"^V"))

test = DataTreatment(df, TreatmentGroup(dims=1, name_expr=r"^V"), TreatmentGroup(dims=2))

# test = DataTreatment(df)

# TreatmentGroup(win=wholewindow(), features=(maximum, minimum, mean))

