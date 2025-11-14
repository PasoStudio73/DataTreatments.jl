# ---------------------------------------------------------------------------- #
#                              custom functions                                #
# ---------------------------------------------------------------------------- #
rootpower(x) = mean(abs2, x) |> sqrt
halfstd(x) = std(x) ./ convert(eltype(x), sqrt(1 - (2 / π)))

# ---------------------------------------------------------------------------- #
#                               core functions                                 #
# ---------------------------------------------------------------------------- #
function _zscore(x::AbstractArray; method::Symbol)
    (y,o) = if method == :std
        (Statistics.mean(x), Statistics.std(x))
    elseif method == :robust
        _y = Statistics.median(x)
        (_y, Statistics.median(abs.(x .- _y)))
    elseif method == :half
        (minimum(x), halfstd(x))
    else
        throw(ArgumentError("method must be :std, :robust or :half, got :$method"))
    end
    (x) -> (x - y) / o
end

function _sigmoid(x::AbstractArray)
    y, o = Statistics.mean(x), Statistics.std(x)
    (x) -> inv(1 + exp(-(x - y) / o))
end

function _pnorm(x::AbstractArray; p::Real)
    x_filtered = filter(!isnan, vec(x))
    s = isempty(x_filtered) ? one(eltype(x)) : LinearAlgebra.norm(x_filtered, p)
    Base.Fix2(/, s)
end

function _scale(x::AbstractArray; factor::Symbol)
    s = if factor == :std
        Statistics.std(x)
    elseif factor == :mad
        StatsBase.mad(x; normalize=false)
    elseif factor == :first
        first(x)
    elseif factor == :iqr
        StatsBase.iqr(x)
    else
        throw(ArgumentError("factor must be :std, :mad, :first, or :iqr, got :$factor"))
    end
    
    Base.Fix2(/, s)
end

function _minmax(x::AbstractArray; lower::Real=0.0, upper::Real=1.0)
    xmin, xmax = extrema(x)    
    scale = (upper - lower) / (xmax - xmin)
    (x) -> clamp(lower + (x - xmin) * scale, lower, upper)
end

_center(y)     = (x) -> x - y

_unitpower(p)  = Base.Fix2(/, p)
_outliersuppress(y, o; thr=5.0) = (x) -> abs(o) > thr * o ? y + sign(x - y) * thr * o : x
function _minmaxclip(l,u)
    (x) -> begin
        l == u && (return x == u ? 0.5 : (x > u) * one(u)) # Return 1.0 if x > u, else 0.0
        return clamp((x - l) / (u - l), 0.0, 1.0)
    end
end

# ---------------------------------------------------------------------------- #
#                              caller functions                                #
# ---------------------------------------------------------------------------- #
"""
    zscore(; method::Symbol=:std) -> Function

Create a z-score normalization function that standardizes data by centering and scaling.

# Arguments
- `method::Symbol=:std`: Method for computing the z-score
  - `:std` (default): Standard z-score using mean and standard deviation
  - `:robust`: Robust z-score using median and median absolute deviation (MAD)
  - `:half`: Half-normal z-score using minimum and half-standard deviation

# Methods

## Standard Z-Score (`:std`)
Centers data to mean 0 and scales to standard deviation 1.
```math
z = \\frac{x - \\mu}{\\sigma}
```
where μ is the mean and σ is the standard deviation.

## Robust Z-Score (:robust)
Centers data to median 0 and scales to median absolute deviation 1.
More resistant to outliers than standard z-score.
```math
z = \\frac{x - \\text{median}(x)}{\\text{MAD}(x)}
```
where MAD is the median absolute deviation.

## Half-Normal Z-Score (:half)
Normalizes to the standard half-normal distribution using minimum and half-standard deviation.
```math
z = \\frac{x - \\min(x)}{\\sigma_{\\text{half}}}
```
where σ_half = σ / √(1 - 2/π).

## Examples
```julia
# Standard z-score normalization
X = rand(100, 50)
X_norm = element_norm(X, zscore())
# Result: mean ≈ 0, std ≈ 1

# Robust z-score (resistant to outliers)
X_robust = element_norm(X, zscore(method=:robust))
# Result: median ≈ 0, MAD ≈ 1

# Half-normal z-score
X_half = element_norm(X, zscore(method=:half))
# Result: minimum ≈ 0, scaled by half-standard deviation

# Tabular normalization (column-wise)
X_tab = tabular_norm(X, zscore())
# Each column: mean ≈ 0, std ≈ 1

# Row-wise normalization
X_row = tabular_norm(X, zscore(); dim=:row)
# Each row: mean ≈ 0, std ≈ 1
```

## References
Standard z-score: https://en.wikipedia.org/wiki/Standard_score
Robust statistics: https://en.wikipedia.org/wiki/Robust_statistics
Half-normal distribution: https://en.wikipedia.org/wiki/Half-normal_distribution
"""
zscore(; method::Symbol=:std)::Function = x -> _zscore(x; method)

"""
    sigmoid() -> Function

Create a sigmoid normalization function that maps data to the interval (0, 1).

The sigmoid (or logistic) function provides a smooth, S-shaped transformation that 
maps the entire real line to the bounded interval (0, 1), with the steepest slope 
at the mean of the data.

# Formula
```math
\\sigma(x) = \\frac{1}{1 + e^{-\\frac{x - \\mu}{\\sigma}}}
```
where:
- μ (mu) is the mean of the input data
- σ (sigma) is the standard deviation of the input data
- The output is bounded: 0 < σ(x) < 1

## Examples
```julia
# Basic sigmoid normalization
X = rand(100, 50)
X_sigmoid = element_norm(X, sigmoid())
# Result: all values in (0, 1), mean(X) → 0.5

# Compare with linear scaling
X_minmax = element_norm(X, minmax())
# minmax: exact [0, 1] bounds with linear mapping
# sigmoid: asymptotic (0, 1) bounds with S-curve

# Handling outliers gracefully
X_outliers = [1, 2, 3, 4, 100]  # One extreme outlier
X_sig = element_norm(X_outliers, sigmoid())
# The outlier (100) is compressed toward 1, not linearly stretched

# Tabular normalization (column-wise)
X_tab = tabular_norm(X, sigmoid())
# Each column: sigmoid transformation with its own mean/std

# Row-wise normalization
X_row = tabular_norm(X, sigmoid(); dim=:row)
# Each row: independent sigmoid transformation
```

## References
Sigmoid function: https://en.wikipedia.org/wiki/Sigmoid_function
Logistic function: https://en.wikipedia.org/wiki/Logistic_function
Feature scaling: https://en.wikipedia.org/wiki/Feature_scaling

"""
sigmoid()::Function = x -> _sigmoid(x)

"""
    norm(; p::Real=2) -> Function

Create a normalization function that scales data by the p-norm.

The p-norm normalization divides each element by the p-norm of the entire dataset,
ensuring that the normalized data has unit p-norm. This is particularly useful for
standardizing data magnitudes across different scales.

# Arguments
- `p::Real=2`: The norm order (default: 2)
  - `p = 1`: Manhattan norm (sum of absolute values)
  - `p = 2`: Euclidean norm (default, root sum of squares)
  - `p = Inf`: Infinity norm (maximum absolute value)
  - `p > 0`: General p-norm

# Formula

## General p-norm (p ≥ 1):
```math
\\|x\\|_p = \\left(\\sum_{i=1}^{n} |x_i|^p\\right)^{1/p}
```
Special cases:
- L1 norm (p=1): `‖x‖₁ = Σ|xᵢ|` (Manhattan/taxicab norm)
- L2 norm (p=2): `‖x‖₂ = √(Σxᵢ²)` (Euclidean norm, default)
- L∞ norm (p=Inf): `‖x‖∞ = max(|xᵢ|)` (Maximum norm)
The normalized value is: `x_normalized = x / ‖x‖ₚ`

## Examples
```julia
# L2 norm (Euclidean, default)
X = rand(100, 50)
X_norm = element_norm(X, norm())
# Result: ‖X‖₂ = 1

# L1 norm (Manhattan)
X_L1 = element_norm(X, norm(p=1))
# Result: sum(abs, X) = 1

# L∞ norm (Maximum)
X_Linf = element_norm(X, norm(p=Inf))
# Result: maximum(abs, X) = 1

# Custom p-norm
X_L4 = element_norm(X, norm(p=4))
# Result: (sum(X.^4))^(1/4) = 1

# Tabular normalization (column-wise)
X_tab = tabular_norm(X, norm())
# Each column: ‖column‖₂ = 1

# Row-wise L∞ normalization
X_row = tabular_norm(X, norm(p=Inf); dim=:row)
# Each row: max(abs, row) = 1

# Unit vector creation
v = [3.0, 4.0]
v_unit = element_norm(v, norm())  # [0.6, 0.8] with ‖v‖₂ = 1

# Signal power normalization
signal = randn(1000)
signal_norm = element_norm(signal, norm(p=2))
# Normalized to unit energy
```

## References
Vector norms: https://en.wikipedia.org/wiki/Norm_(mathematics)
Lp spaces: https://en.wikipedia.org/wiki/Lp_space
Unit vector: https://en.wikipedia.org/wiki/Unit_vector
Feature scaling: https://en.wikipedia.org/wiki/Feature_scaling
"""
pnorm(; p::Real=2)::Function = x -> _norm(x; p)

"""
    scale(; factor::Symbol=:std) -> Function

Create a normalization function that scales data by a specified scale factor.

Scale normalization divides data by a characteristic scale measure, standardizing
the spread or magnitude without necessarily centering the data. This is useful when
you want to normalize variability but preserve the mean or baseline.

# Arguments
- `factor::Symbol=:std`: Scale factor to use (default: `:std`)
  - `:std`: Scale by standard deviation
  - `:mad`: Scale by median absolute deviation
  - `:first`: Scale by the first element value
  - `:iqr`: Scale by interquartile range

# Scale Factor Options

## Standard Deviation (`:std`, default)
Scale data to have standard deviation of 1.
```math
x_{\\text{scaled}} = \\frac{x}{\\sigma}
```
where σ is the standard deviation.

## Median Absolute Deviation (:mad)
Scale data to have median absolute deviation of 1.
```math
x_{\\text{scaled}} = \\frac{x}{\\text{MAD}(x)}
```
where MAD = median(|x - median(x)|).

## First Element (:first)
Scale data by the value of the first element.
```math
x_{\\text{scaled}} = \\frac{x}{x_1}
```

## Interquartile Range (:iqr)
Scale data to have interquartile range of 1.
```math
x_{\\text{scaled}} = \\frac{x}{\\text{IQR}(x)}
```
where IQR = Q₃ - Q₁ (75th percentile - 25th percentile).

## Examples
```julia
# Standard deviation scaling (default)
X = rand(100, 50)
X_scaled = element_norm(X, scale())
# Result: std(X_scaled) ≈ 1, mean unchanged

# Robust scaling with MAD
X_outliers = [1, 2, 3, 4, 100]  # Has outlier
X_mad = element_norm(X_outliers, scale(factor=:mad))
# More robust than std scaling

# IQR scaling (robust to extreme values)
X_iqr = element_norm(X, scale(factor=:iqr))
# Result: IQR(X_iqr) ≈ 1

# Baseline normalization (first element)
prices = [100.0, 105.0, 98.0, 110.0]
prices_norm = element_norm(prices, scale(factor=:first))
# Result: [1.0, 1.05, 0.98, 1.10] - percentage of initial price

# Tabular normalization (column-wise)
X_tab = tabular_norm(X, scale())
# Each column: std = 1

# Row-wise scaling with IQR
X_row = tabular_norm(X, scale(factor=:iqr); dim=:row)
# Each row: IQR = 1

# Time series baseline normalization
timeseries = rand(1000)
ts_norm = element_norm(timeseries, scale(factor=:first))
# Normalized to first observation = 1.0
```

## References
- Standard deviation: https://en.wikipedia.org/wiki/Standard_deviation
- Median absolute deviation: https://en.wikipedia.org/wiki/Median_absolute_deviation
- Interquartile range: https://en.wikipedia.org/wiki/Interquartile_range
"""
scale(; factor::Symbol=:std) = x -> _scale(x; factor)

"""
    minmax(; lower::Real=0.0, upper::Real=1.0) -> Function

Create a min-max normalization function that rescales data to a specified range.

Min-max normalization (also known as feature scaling) linearly transforms data from 
its original range [min(x), max(x)] to a target range [lower, upper].

# Arguments
- `lower::Real=0.0`: Lower bound of the output range (default: 0.0)
- `upper::Real=1.0`: Upper bound of the output range (default: 1.0)

# Formula
```math
x_{\\text{scaled}} = \\text{lower} + \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}} \\cdot (\\text{upper} - \\text{lower})
```
This maps the original range [x_min, x_max] to [lower, upper] via affine transformation.

## Examples
```julia
# Standard min-max scaling to [0, 1]
X = rand(100, 50)
X_norm = element_norm(X, minmax())
# Result: min ≈ 0, max ≈ 1

# Custom range scaling to [-1, 1]
X_scaled = element_norm(X, minmax(lower=-1.0, upper=1.0))
# Result: min ≈ -1, max ≈ 1

# Scale to [0, 100] (percentage-like)
X_percent = element_norm(X, minmax(lower=0.0, upper=100.0))
# Result: min ≈ 0, max ≈ 100

# Tabular normalization (column-wise)
X_tab = tabular_norm(X, minmax())
# Each column scaled independently to [0, 1]

# Row-wise normalization to custom range
X_row = tabular_norm(X, minmax(lower=-5.0, upper=5.0); dim=:row)
# Each row scaled independently to [-5, 5]
```

## References
Feature scaling: https://en.wikipedia.org/wiki/Feature_scaling
Min-max normalization: https://en.wikipedia.org/wiki/Normalization_(statistics)
"""
minmax(; lower::Real=0.0, upper::Real=1.0)::Function = x -> _minmax(x; lower, upper)

center()::Function          = x -> _center(Statistics.mean(x))

unitpower()::Function       = x -> _unitpower(rootpower(x))
outliersuppress()::Function = x -> _outliersuppress(Statistics.mean(x), Statistics.std(x))
minmaxclip()::Function      = x -> _minmaxclip(minimum(x), maximum(x))



# ---------------------------------------------------------------------------- #
#                              normalize functions                             #
# ---------------------------------------------------------------------------- #
"""
    element_norm(X::AbstractArray, n::Base.Callable) -> AbstractArray

Normalize a single array element using global statistics computed across all elements.

# Arguments
- `X::AbstractArray`: Input array of any dimension (vector, matrix, tensor, etc.)
- `n::Base.Callable`: Normalization function constructor that computes parameters from data

# Supported Normalization Methods
- `zscore()`: Z-score normalization using mean and standard deviation
- `sigmoid()`: Sigmoid transformation using mean and standard deviation  
- `minmax()`: Min-max scaling to [0, 1] range
- `center()`: Mean centering (subtract mean)
- `unitenergy()`: Scale by root sum of squares
- `unitpower()`: Scale by root mean square
- `halfzscore()`: Z-score using minimum and half-standard deviation
- `outliersuppress()`: Suppress outliers beyond threshold
- `minmaxclip()`: Min-max scaling with clipping to [0, 1]

# Returns
- `AbstractArray`: Normalized array with same shape as input

# Examples
```julia
X = rand(100, 50)
X_norm = element_norm(X, zscore())      # Z-score normalization
X_norm = element_norm(X, minmax())     # Min-max scaling
X_norm = element_norm(X, center())      # Mean centering
```
"""
function element_norm(X::AbstractArray{T}, n::Base.Callable)::AbstractArray where {T<:AbstractFloat}
    _X = Iterators.flatten(X)
    nfunc = n(collect(_X))
    [nfunc(X[idx]) for idx in CartesianIndices(X)]
end
element_norm(X::AbstractArray{T}, args...) where {T<:Real} = element_norm(Float64.(X), args...)

function tabular_norm(
    X::AbstractArray{T},
    n::Base.Callable;
    dim::Symbol=:col
)::AbstractArray where {T<:AbstractFloat}
    dim in (:col, :row) || throw(ArgumentError("dim must be :col or :row, got :$dim"))

    dim == :row && (X = X')
    cols = Iterators.flatten.(eachcol(X))
    nfuncs = @inbounds [n(collect(cols[i])) for i in eachindex(cols)]
    dim == :row ? [nfuncs[idx[2]](X[idx]) for idx in CartesianIndices(X)]' :
                  [nfuncs[idx[2]](X[idx]) for idx in CartesianIndices(X)]
end
tabular_norm(X::AbstractArray{T}, args...; kwargs...) where {T<:Real} = 
    tabular_norm(Float64.(X), args...;kwargs...)

@inline function _ds_norm!(Xn::AbstractArray, X::AbstractArray, nfunc)
    @inbounds @simd for i in eachindex(X, Xn)
        Xn[i] = nfunc(X[i])
    end
    return Xn
end

function ds_norm(X::AbstractArray{T}, n::Base.Callable)::AbstractArray where {T<:AbstractArray{<:AbstractFloat}}
    # compute normalization functions for each column
    cols = Iterators.flatten.(eachcol(X))
    nfuncs = Vector{Function}(undef, length(cols))
    Threads.@threads for i in axes(X, 2)
        nfuncs[i] = n(collect(cols[i]))
    end
    
    # apply normalization
    Xn = similar(X)
    Threads.@threads for j in axes(X, 2)
        @inbounds for i in axes(X, 1)
            Xn[i, j] = similar(X[i, j])
            _ds_norm!(Xn[i, j], X[i, j], nfuncs[j])
        end
    end
    
    return Xn
end
ds_norm(X::AbstractArray{T}, args...) where {T<:AbstractArray{<:Real}} = ds_norm(Float64.(X), args...)