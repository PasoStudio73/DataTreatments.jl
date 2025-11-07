# ---------------------------------------------------------------------------- #
#                              min/max normalize                               #
# ---------------------------------------------------------------------------- #
minmax_normalize(c, args...; kwars...) = minmax_normalize!(deepcopy(c), args...; kwars...)

"""
    minmax_normalize!(X; kwargs...)
    minmax_normalize!(X, min::Real, max::Real)

Apply min-max normalization to scale values to the range [0,1], modifying the input in-place.

# Common Methods
- `minmax_normalize!(X::AbstractMatrix; kwargs...)`: Normalize a matrix
- `minmax_normalize!(df::AbstractDataFrame; kwargs...)`: Normalize a DataFrame
- `minmax_normalize!(md::MultiData.MultiDataset, frame_index::Integer; kwargs...)`: Normalize a specific frame in a multimodal dataset
- `minmax_normalize!(v::AbstractArray{<:Real}, min::Real, max::Real)`: Normalize an array using specific min/max values
- `minmax_normalize!(v::AbstractArray{<:AbstractArray{<:Real}}, min::Real, max::Real)`: Normalize an array of arrays

# Arguments
- `X`: The data to normalize (matrix, DataFrame, or MultiDataset)
- `frame_index`: For MultiDataset, the index of the frame to normalize
- `min::Real`: Minimum value for normalization (when provided directly)
- `max::Real`: Maximum value for normalization (when provided directly)

# Keyword Arguments
- `min_quantile::Real=0.0`: Lower quantile threshold for normalization
  - `0.0`: Use the absolute minimum (no outlier exclusion)
  - `> 0.0`: Use the specified quantile as minimum (e.g., 0.05 excludes bottom 5% as outliers)
- `max_quantile::Real=1.0`: Upper quantile threshold for normalization
  - `1.0`: Use the absolute maximum (no outlier exclusion)
  - `< 1.0`: Use the specified quantile as maximum (e.g., 0.95 excludes top 5% as outliers)
- `col_quantile::Bool=true`: How to calculate quantiles
  - `true`: Calculate separate quantiles for each column (column-wise normalization)
  - `false`: Calculate global quantiles across the entire dataset

# Returns
The input data, normalized in-place.

# Throws
- `DomainError`: If min_quantile < 0, max_quantile > 1, or max_quantile ≤ min_quantile

# Details
## Matrix/DataFrame normalization:
When normalizing matrices or DataFrames, this function:
1. Validates the quantile parameters
2. Determines min/max values based on the specified quantiles
3. If `col_quantile=true`, calculates separate min/max for each column
4. If `col_quantile=false`, uses the same min/max across the entire dataset
5. Applies the normalization to transform values to the [0,1] range

## Array normalization:
For direct array normalization with provided min/max values:
1. If min equals max, returns an array filled with 0.5 values
2. Otherwise, scales values to [0,1] range using the formula: (x - min) / (max - min)
"""
# function minmax_normalize!(
#     md::MultiData.MultiDataset,
#     frame_index::Integer;
#     min_quantile::Real = 0.0,
#     max_quantile::Real = 1.0,
#     col_quantile::Bool = true,
# )
#     return minmax_normalize!(
#         MultiData.modality(md, frame_index);
#         min_quantile = min_quantile,
#         max_quantile = max_quantile,
#         col_quantile = col_quantile
#     )
# end

function minmax_normalize!(
    X::AbstractMatrix;
    min_quantile::Real = 0.0,
    max_quantile::Real = 1.0,
    col_quantile::Bool = true,
)
    min_quantile < 0.0 &&
        throw(DomainError(min_quantile, "min_quantile must be greater than or equal to 0"))
    max_quantile > 1.0 &&
        throw(DomainError(max_quantile, "max_quantile must be less than or equal to 1"))
    max_quantile ≤ min_quantile &&
        throw(DomainError("max_quantile must be greater then min_quantile"))

    icols = eachcol(X)

    if (!col_quantile)
        # look for quantile in entire dataset
        itdf = Iterators.flatten(Iterators.flatten(icols))
        min = StatsBase.quantile(itdf, min_quantile)
        max = StatsBase.quantile(itdf, max_quantile)
    else
        # quantile for each column
        itcol = Iterators.flatten.(icols)
        min = StatsBase.quantile.(itcol, min_quantile)
        max = StatsBase.quantile.(itcol, max_quantile)
    end
    minmax_normalize!.(icols, min, max)
    return X
end

function minmax_normalize!(
    df::AbstractDataFrame;
    kwargs...
)
    minmax_normalize!(Matrix(df); kwargs...)
end

function minmax_normalize!(
    v::AbstractArray{<:AbstractArray{<:Real}},
    min::Real,
    max::Real
)
    return minmax_normalize!.(v, min, max)
end

function minmax_normalize!(
    v::AbstractArray{<:Real},
    min::Real,
    max::Real
)
    if (min == max)
        return repeat([0.5], length(v))
    end
    min = float(min)
    max = float(max)
    max = 1 / (max - min)
    rt = StatsBase.UnitRangeTransform(1, 1, true, [min], [max])
    # This function doesn't accept Integer
    return StatsBase.transform!(rt, v)
end