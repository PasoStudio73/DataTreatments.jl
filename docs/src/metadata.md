```@meta
CurrentModule = DataTreatments
```
# [Metadata](@id metadata)

The metadata module defines the core types used to describe individual columns
(features) inside a [`DataTreatment`](@ref). Each struct captures **what kind of
data** a column contains together with **validity information** (which rows are
valid, missing, or `NaN`).

There are four concrete subtypes of `AbstractDataFeature`:

| Type | Use case |
|:-----|:---------|
| [`DiscreteFeat`](@ref) | Categorical / discrete columns (e.g., labels, classes) |
| [`ContinuousFeat`](@ref) | Scalar numeric columns (e.g., measurements, sensor readings) |
| [`AggregateFeat`](@ref) | Multidimensional columns flattened via windowed aggregation |
| [`ReduceFeat`](@ref) | Multidimensional columns whose size is reduced while preserving dimensionality |

---

## Types

### DiscreteFeat

```@docs
DiscreteFeat
```

### ContinuousFeat

```@docs
ContinuousFeat
```

### AggregateFeat

```@docs
AggregateFeat
```

### ReduceFeat

```@docs
ReduceFeat
```

---

## Getter methods

All feature types share a common set of accessor functions defined on
`AbstractDataFeature`. Additional getters are available only for specific
subtypes.

### Common getters (all subtypes)

```@docs
get_id
get_idx
get_vname
get_valididxs(::AbstractDataFeature)
get_missingidxs(::AbstractDataFeature)
```

### Numeric getters (`ContinuousFeat`, `AggregateFeat`, `ReduceFeat`)

```@docs
get_nanidxs(::Union{ContinuousFeat,AggregateFeat,ReduceFeat})
```

### Multidimensional getters (`AggregateFeat`, `ReduceFeat`)

```@docs
get_dims(::Union{AggregateFeat,ReduceFeat})
get_hasmissing(::Union{AggregateFeat,ReduceFeat})
get_hasnans(::Union{AggregateFeat,ReduceFeat})
```

### Discrete-only getters (`DiscreteFeat`)

```@docs
get_levels
```

### Aggregate-only getters (`AggregateFeat`)

```@docs
get_feat
get_nwin
```

### Reduce-only getters (`ReduceFeat`)

```@docs
get_reducefunc
```