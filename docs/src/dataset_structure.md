```@meta
CurrentModule = DataTreatments
```

# Data Structures

This page describes the complete type hierarchy used to represent
processed datasets in `DataTreatments.jl`, from low-level metadata
features up to the top-level `DataTreatment` container, and how
[`load_dataset`](@ref) drives the full pipeline.

---

## Type Hierarchy Overview

```
AbstractDataFeature
├── DiscreteFeat{T}       # metadata for a categorical column
├── ContinuousFeat{T}     # metadata for a numeric scalar column
├── AggregateFeat{T}      # metadata for a ts → scalar aggregation
└── ReduceFeat{T}         # metadata for a ts → smaller array reduction

AbstractDataset
├── DiscreteDataset{T}       # integer-coded categorical matrix
├── ContinuousDataset{T}     # float matrix
└── MultidimDataset{T, S}
     ├── S = AggregateFeat   →  tabular scalar output
     └── S = ReduceFeat      →  reduced array output

DataTreatment{T}          # top-level container (load_dataset output)
├── data    ::Vector{AbstractDataset}
├── target  ::AbstractVector
├── treats  ::Vector{TreatmentGroup}
└── balance ::Union{Nothing, AbstractBalance,
                    Tuple{Vararg{<:AbstractBalance}}}
```

---

### Supported input sources

| Dispatch | Input | Notes |
|:---------|:------|:------|
| Dispatch 1 | `AbstractMatrix` | core method |
| Dispatch 2 | `DataFrame` | extracts matrix + column names |
| Dispatch 3 | `DataTreatment` | re-treats existing output |

### Full pipeline

```
data / df / dt
      │
      ▼  _inspecting(data)
      │   → per-column metadata (type, missing, NaN, dims)
      │
      ▼  encode target  →  CategoricalVector
      │
      ▼  for each TreatmentGroup:
      │   ├─ classify → discrete_ids, continuous_ids, multidim_ids
      │   │
      │   ├─ DiscreteDataset(discrete_ids, ...)
      │   │    └─ _discrete_encode  →  integer-coded matrix
      │   │
      │   ├─ ContinuousDataset(continuous_ids, ...)
      │   │    └─ cast to float_type  →  float matrix
      │   │
      │   └─ MultidimDataset(multidim_ids, ..., aggrfunc)
      │        ├─ aggregate(...)  →  scalar cols (AggregateFeat)
      │        └─ reducesize(...) →  array cols  (ReduceFeat)
      │
      ▼  if balance provided:
      │   └─ for each dataset:
      │       (data, target) = bal₁ ∘ … ∘ balₙ(data, target)
      │
      └─ DataTreatment{float_type}(ds, target, treats, balance)
```

### Re-treating an existing `DataTreatment`

```julia
dt2 = load_dataset(dt, TreatmentGroup(dims=1, impute=(LOCF(),)))
```

Internally reconstructs the raw matrix from `dt.data` and
`get_target(dt)`, then runs the full pipeline with the new
treatment groups.

---

## Treatment Directives

### `TreatmentGroup`

```@docs
TreatmentGroup
```

#### Column selection filters

```
All columns in the dataset
        │
        ├─ dims filter
        │   keep columns with array length == dims
        │   (skip if dims == -1)
        │
        ├─ datatype filter
        │   :discrete   → !(T <: Float) && !(T <: AbstractArray)
        │   :continuous → T <: Float
        │   :multidim   → T <: AbstractArray
        │   :all        → any type (skip filter)
        │
        └─ name_expr filter
            Regex    → match(name_expr, name) !== nothing
            Callable → name_expr(name)
            Vector   → name ∈ name_expr
            (skip if name_expr == r".*")
                    │
                    ▼
            ids ::Vector{Int}
```

#### From directive to output datasets

```
TreatmentGroup
│   ids = [1, 3, 5, 7, 8]
│
├─ discrete ids   → [1, 3]
│   └─▶ DiscreteDataset{T}
│        ├── data :: Matrix{Union{Int,Missing}}
│        └── info :: Vector{DiscreteFeat}
│
├─ continuous ids → [5]
│   └─▶ ContinuousDataset{T}
│        ├── data :: Matrix{Union{Missing,T}}
│        └── info :: Vector{ContinuousFeat}
│
└─ multidim ids  → [7, 8]
    │
    ├─ aggrfunc = aggregate(...)
    │   └─▶ MultidimDataset{T, AggregateFeat}
    │        ├── data   :: Matrix{T}
    │        ├── info   :: Vector{AggregateFeat}
    │        └── groups :: Vector{Vector{Int}}
    │
    └─ aggrfunc = reducesize(...)
        └─▶ MultidimDataset{T, ReduceFeat}
             ├── data   :: Matrix{AbstractArray{T}}
             ├── info   :: Vector{ReduceFeat}
             └── groups :: nothing
```

#### Multiple `TreatmentGroup`s

```julia
dt = load_dataset(df, y, t1, t2, t3)
```

```
t1 = TreatmentGroup(datatype=:continuous, norm=MinMax)
      └─▶ ContinuousDataset

t2 = TreatmentGroup(name_expr=r"^signal_",
                    aggrfunc=aggregate(...), groupby=:vname)
      └─▶ MultidimDataset{T, AggregateFeat}

t3 = TreatmentGroup(name_expr=r"^audio_",
                    aggrfunc=reducesize(256))
      └─▶ MultidimDataset{T, ReduceFeat}

DataTreatment.data = [
    ContinuousDataset,             # from t1
    MultidimDataset{AggregateFeat},# from t2
    MultidimDataset{ReduceFeat},   # from t3
]
```

#### Class balancing

Balancing is applied after all datasets are built, once per
dataset:

```julia
# single strategy
dt = load_dataset(df, y, t1; balance=SMOTE(k=5))

# chained strategies
dt = load_dataset(df, y, t1;
    balance=(SMOTE(k=5), TomekUndersampler()))
```

See [Imbalance](@ref imbalance) for available strategies.

---

## Metadata Feature Structs

One struct is created per output column and carries diagnostic
information (valid/missing/NaN indices, source column id, etc.).

### `DiscreteFeat{T}`

```@docs
DiscreteFeat
```

---

### `ContinuousFeat{T}`

```@docs
ContinuousFeat
```

---

### `AggregateFeat{T}`

```@docs
AggregateFeat
```

---

### `ReduceFeat{T}`

```@docs
ReduceFeat
```

---

## Dataset Structs

Each dataset wraps a processed data matrix together with a `Vector`
of the corresponding metadata feature structs.

### `DiscreteDataset{T}`

```@docs
DiscreteDataset
```

---

### `ContinuousDataset{T}`

```@docs
ContinuousDataset
```

---

### `MultidimDataset{T,S}`

```@docs
MultidimDataset
```

---

## Top-level Container: `DataTreatment{T}`

```@docs
DataTreatment
```

---

## Accessor Reference

| Method | Returns |
|:-------|:--------|
| `get_discrete(dt)` | `(Matrix, vnames)` for categorical cols |
| `get_continuous(dt)` | `(Matrix{T}, vnames)` for scalar cols |
| `get_aggregated(dt)` | `(Matrix{T}, vnames)` for aggregated series |
| `get_reduced(dt)` | `(Matrix{Array{T}}, vnames)` for series |
| `get_tabular(dt)` | merged tabular matrix + all column names |
| `get_multidim(dt)` | reduced multidim matrix + column names |
| `get_target(dt)` | the encoded target vector |
| `get_treats(dt)` | the applied `TreatmentGroup` list |
| `get_balance(dt)` | the balancing strategy or `nothing` |