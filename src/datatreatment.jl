# ---------------------------------------------------------------------------- #
#                                     types                                    #
# ---------------------------------------------------------------------------- #
const DefaultAggrFunc = aggregate(win=(wholewindow(),), features=(maximum, minimum, mean))
const DefaultGrouped = false
const DefaultTreatmentGroup = TreatmentGroup(aggrfunc=DefaultAggrFunc, grouped=DefaultGrouped)

# ---------------------------------------------------------------------------- #
#                             internal functions                               #
# ---------------------------------------------------------------------------- #
"""
    _build_datasets(
        id::Vector,
        data::Matrix,
        ds_struct::DataStructure,
        idxs::Vector{Int},
        treat::TreatmentGroup;
        float_type::Type=Float64
    ) -> (ds_td, ds_tc, ds_md)

Internal function that partitions the selected columns of a dataset into three
macro categories based on their data type, and stores each partition in the
appropriate structure:

- **Discrete data**: columns whose element type is neither `AbstractFloat` nor
  `AbstractArray` (e.g., categorical labels, strings, integers). Stored in a
  [`DiscreteDataset`](@ref).
- **Continuous data**: columns whose element type is a subtype of `AbstractFloat`
  (scalar numeric values). Stored in a [`ContinuousDataset`](@ref).
- **Multidimensional data**: columns whose element type is a subtype of
  `AbstractArray` (e.g., time series, images, spectrograms). Stored in a
  [`MultidimDataset`](@ref), processed according to the provided `treat`'s
  aggregation function.

Columns whose detected type is `nothing` are silently skipped.

# Arguments
- `id::Vector`: An identifier tag for the dataset partition (e.g.,
  `[:treatment_group, 1]`), propagated to each sub-dataset for traceability.
- `data::Matrix`: The raw data matrix.
- `ds_struct::DataStructure`: Precomputed metadata about the dataset
  (types, dimensions, validity indices). See [`DataStructure`](@ref).
- `idxs::Vector{Int}`: Column indices to consider (typically from a
  [`TreatmentGroup`](@ref)).
- `treat::TreatmentGroup`: The treatment group containing the aggregation or
  reduction function and optional groupby information.
- `float_type::Type`: The floating-point type for numeric output
  (default: `Float64`).

# Returns
A tuple of three elements `(ds_td, ds_tc, ds_md)`:
- `ds_td`: A [`DiscreteDataset`](@ref) or `nothing` if no discrete
  columns are present.
- `ds_tc`: A [`ContinuousDataset`](@ref) or `nothing` if no
  continuous columns are present.
- `ds_md`: A [`MultidimDataset`](@ref) or `nothing` if no
  multidimensional columns are present.

See also: [`DataTreatment`](@ref), [`TreatmentGroup`](@ref),
[`DiscreteDataset`](@ref), [`ContinuousDataset`](@ref),
[`MultidimDataset`](@ref)
"""
function _build_datasets(
    id::Vector,
    data::Matrix,
    ds_struct::DataStructure,
    idxs::Vector{Int},
    treat::TreatmentGroup;
    float_type::Type=Float64
)
    aggrfunc = get_aggrfunc(treat);
    valtype = get_datatype(ds_struct)
    groups = has_groupby(treat) ? get_groupby(treat) : nothing

    td_cols = idxs ∩ findall(T -> !isnothing(T) && !(T <: AbstractFloat) && !(T <: AbstractArray), valtype)
    tc_cols = idxs ∩ findall(T -> !isnothing(T) && T <: AbstractFloat, valtype)
    md_cols = idxs ∩ findall(T -> !isnothing(T) && T <: AbstractArray, valtype)

    ds_td = isempty(td_cols) ?
        nothing :
        DiscreteDataset(id, data, ds_struct, td_cols)
    ds_tc = isempty(tc_cols) ?
        nothing :
        ContinuousDataset(id, data, ds_struct, tc_cols, float_type)
    ds_md = isempty(md_cols) ?
        nothing :
        MultidimDataset(id, data, ds_struct, md_cols, aggrfunc, float_type, groups)

    return ds_td, ds_tc, ds_md
end

"""
    _get_treatments_datasets(dt::DataTreatment, treats::Vector{<:TreatmentGroup}) -> Vector{AbstractDataset}

Extract the processed datasets using the [`TreatmentGroup`](@ref) directives
specified by the user when calling [`get_dataset`](@ref).

Returns a flat `Vector{AbstractDataset}` where each element is one of:
- [`DiscreteDataset`](@ref): columns with categorical/discrete data.
- [`ContinuousDataset`](@ref): columns with scalar numeric data.
- [`MultidimDataset`](@ref): columns with array-valued data (e.g., time series,
  spectrograms), processed according to the aggregation or reduction function
  specified in each treatment group.

When multidimensional columns originate from arrays of different dimensionalities
(e.g., 1D and 2D), they are automatically split into separate `MultidimDataset`s,
one per unique dimensionality. If the user specifies a finer partitioning through
multiple `TreatmentGroup`s, this partitioning is preserved in the output.

!!! note
    Only the columns covered by the user-defined treatment groups are returned.
    Columns not assigned to any `TreatmentGroup` are **not** included; use
    [`_get_leftover_datasets`](@ref) to retrieve them, or [`get_dataset`](@ref)
    to obtain both.

# Arguments
- `dt::DataTreatment`: The `DataTreatment` object containing the raw dataset.
- `treats::Vector{<:TreatmentGroup}`: The instantiated treatment groups specifying
  which columns to select and how to process them.

# Returns
A `Vector{AbstractDataset}` containing, in order:
1. All [`DiscreteDataset`](@ref)s (one per treatment group that has discrete columns).
2. All [`ContinuousDataset`](@ref)s (one per treatment group that has continuous columns).
3. All [`MultidimDataset`](@ref)s, split by source dimensionality.

Empty categories are omitted from the result.

See also: [`DataTreatment`](@ref), [`TreatmentGroup`](@ref),
[`_get_leftover_datasets`](@ref), [`get_dataset`](@ref),
[`_build_datasets`](@ref), [`_split_md_by_dims`](@ref)
"""
function _get_treatments_datasets(dt::DataTreatment, treats::Vector{<:TreatmentGroup})
    idxs = get_idxs(treats)

    data = get_data(dt)
    ds_struct = get_ds_struct(dt)
    float_type = get_float_type(dt)

    ntreats = length(treats)
    ds_td = Vector{Union{Nothing,DiscreteDataset}}(undef, ntreats)
    ds_tc = Vector{Union{Nothing,ContinuousDataset}}(undef, ntreats)
    ds_md = Vector{Union{Nothing,MultidimDataset}}(undef, ntreats)

    Threads.@threads for i in eachindex(treats)
        ds_td[i], ds_tc[i], ds_md[i] = _build_datasets(
            [:treatment_group, i],
            data,
            ds_struct,
            idxs[i],
            treats[i];
            float_type
        )
    end

    td_filtered = filter(!isnothing, ds_td)
    tc_filtered = filter(!isnothing, ds_tc)
    md_filtered = filter(!isnothing, ds_md)

    md_split = isempty(md_filtered) ? AbstractDataset[] : reduce(vcat, _split_md_by_dims.(md_filtered))

    return AbstractDataset[td_filtered; tc_filtered; md_split]
end
"""
    _get_leftover_datasets(dt::DataTreatment, treats::Vector{<:TreatmentGroup}) -> Vector{AbstractDataset}

Complement of [`_get_treatments_datasets`](@ref): returns the dataset columns that
were **not** selected by any user-defined [`TreatmentGroup`](@ref), formatted as
a flat `Vector{AbstractDataset}`.

Leftover columns are partitioned by data type using the same logic as
[`_get_treatments_datasets`](@ref):
- Categorical/discrete columns → [`DiscreteDataset`](@ref)
- Scalar numeric columns → [`ContinuousDataset`](@ref)
- Array-valued columns → [`MultidimDataset`](@ref), processed with the default
  aggregation function (`maximum`, `minimum`, `mean` over the whole window) and
  split by source dimensionality.

!!! note
    If every column of the dataset is covered by the user-defined treatment groups,
    the returned vector will be empty.

# Arguments
- `dt::DataTreatment`: The `DataTreatment` object containing the raw dataset.
- `treats::Vector{<:TreatmentGroup}`: The instantiated treatment groups. Columns
  already assigned to these groups are excluded from the leftover set.

# Returns
A `Vector{AbstractDataset}` containing, in order:
1. A [`DiscreteDataset`](@ref) for leftover categorical columns (if any).
2. A [`ContinuousDataset`](@ref) for leftover scalar numeric columns (if any).
3. One or more [`MultidimDataset`](@ref)s for leftover array-valued columns (if any),
   split by source dimensionality.

Empty categories are omitted from the result.

See also: [`DataTreatment`](@ref), [`_get_treatments_datasets`](@ref),
[`get_dataset`](@ref), [`_build_datasets`](@ref), [`_split_md_by_dims`](@ref)
"""
function _get_leftover_datasets(dt::DataTreatment, treats::Vector{<:TreatmentGroup})
    idxs = setdiff(collect(eachindex(dt)), reduce(vcat, get_idxs(treats)))

    data = get_data(dt)
    ds_struct = get_ds_struct(dt)
    float_type = get_float_type(dt)

    ds_td, ds_tc, ds_md = _build_datasets(
        [:leftover, 1],
        data,
        ds_struct,
        idxs,
        TreatmentGroup(get_ds_struct(dt); aggrfunc=DefaultAggrFunc);
        float_type
    )

    td_filtered = isnothing(ds_td) ? AbstractDataset[] : AbstractDataset[ds_td]
    tc_filtered = isnothing(ds_tc) ? AbstractDataset[] : AbstractDataset[ds_tc]
    md_split = isnothing(ds_md) ? AbstractDataset[] : _split_md_by_dims(ds_md)

    return AbstractDataset[td_filtered; tc_filtered; md_split]
end

# ---------------------------------------------------------------------------- #
#                             custom lazy methods                              #
# ---------------------------------------------------------------------------- #
"""
    get_dataset(
        dt::DataTreatment,
        treatments::Base.Callable...;
        treatment_ds=true,
        leftover_ds=true,
    ) -> (Vector{AbstractDataset}, Vector{TreatmentGroup})

The core function of the DataTreatments package.
Lazily extracts processed datasets from a `DataTreatment` object according to user-specified 
treatment groups and options.

# Purpose

`get_dataset` enables flexible, on-demand extraction of datasets by applying 
user-defined filters and grouping logic (via `TreatmentGroup` and related callables). 
It supports extracting only the columns and transformations the user specifies, 
while also allowing retrieval of any leftover (unassigned) columns.

This is a convenience method that concatenates the results of
[`_get_treatments_datasets`](@ref) and [`_get_leftover_datasets`](@ref). The returned
vector contains, in order:
1. All datasets produced by [`_get_treatments_datasets`](@ref) (discrete, continuous,
   and multidimensional columns covered by user-defined [`TreatmentGroup`](@ref)s).
2. All datasets produced by [`_get_leftover_datasets`](@ref) (columns not assigned to
   any treatment group, processed with the default aggregation function).

Each element is one of:
- [`DiscreteDataset`](@ref): columns with categorical/discrete data.
- [`ContinuousDataset`](@ref): columns with scalar numeric data.
- [`MultidimDataset`](@ref): columns with array-valued data, split by source
  dimensionality.

# Arguments

- `dt::DataTreatment`: The container holding the raw dataset and all metadata.
- `treatments::Base.Callable...`: One or more treatment group callables (e.g., from `TreatmentGroup`) 
  that define how to filter, group, and process columns. Defaults to a single `TreatmentGroup`
  with the default aggregation function (`maximum`, `minimum`, `mean` over the whole window).
- `treatment_ds::Bool=true`: If `true`, include datasets defined by the treatment groups.
- `leftover_ds::Bool=true`: If `true`, include datasets for columns not assigned to any treatment group.

# Returns

A tuple:
- `Vector{AbstractDataset}`: The processed datasets (discrete, continuous, multidimensional).
- `Vector{TreatmentGroup}`: The instantiated treatment groups used for extraction.

# Example

```julia-repl
using DataTreatments, DataFrames, Statistics
```

```@example
df = DataFrame(
    str_col  = [missing, "blue", "green", "red", "blue"],
    sym_col  = [:circle, :square, :triangle, :square, missing],
    int_col  = Int[10, 20, 30, 40, 50],
    V1 = [NaN, missing, 3.0, 4.0, 5.6],
    V2 = [2.5, missing, 4.5, 5.5, NaN],
    ts1 = [NaN, collect(2.0:7.0), missing, collect(4.0:9.0), collect(5.0:10.0)],
    ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), NaN],
)

dt = DataTreatment(df; float_type=Float32)
```

```@example
# With default treatment (aggregate with max, min, mean over whole window)
datasets, treats = get_dataset(dt)
```

```@example
# With custom treatment groups
datasets, treats = get_dataset(dt, 
    TreatmentGroup(dims=0), 
    TreatmentGroup(dims=1, aggrfunc=aggregate(features=(mean, std)))
)
```

```@example
# Only leftover columns
datasets, treats = get_dataset(dt, TreatmentGroup(dims=1); treatment_ds=false)
```

See also: [`DataTreatment`](@ref), [`TreatmentGroup`](@ref), [`DataStructure`](@ref),
[`_get_treatments_datasets`](@ref), [`_get_leftover_datasets`](@ref)
"""
function get_dataset(
    dt::DataTreatment,
    treatments::Vararg{Base.Callable}=DefaultTreatmentGroup;
    treatment_ds::Bool=true,
    leftover_ds::Bool=true,
)
    treats = [treat(get_ds_struct(dt)) for treat in treatments]

    ds = AbstractDataset[]

    treatment_ds && append!(ds, _get_treatments_datasets(dt, treats))
    leftover_ds && append!(ds, _get_leftover_datasets(dt, treats))

    return ds, treats
end

# ---------------------------------------------------------------------------- #
#                                 show method                                  #
# ---------------------------------------------------------------------------- #
function Base.show(io::IO, dt::DataTreatment)
    nrows, ncols = size(dt.data)
    target_info = isnothing(dt.target) ? "none" : "$(length(dt.target)) labels"
    print(io, "DataTreatment($(nrows)×$(ncols), target=$(target_info), float_type=$(dt.float_type))")
end

function Base.show(io::IO, ::MIME"text/plain", dt::DataTreatment)
    nrows, ncols = size(dt.data)
    target_info = isnothing(dt.target) ? "Unsupervised" : "Supervised"

    println(io, "DataTreatment")
    println(io, "  Data size:    $(nrows) × $(ncols)")
    println(io, "  Target:       ", target_info)
    println(io, "  Float type:   $(dt.float_type)")
    print(io,   "  DS structure: $(dt.ds_struct)")
end