```@example
using DataTreatments
using DataFrames
using CategoricalArrays
```

```@example
# ---------------------------------------------------------------------------- #
#                     dataset with only discrete features                      #
# ---------------------------------------------------------------------------- #
df = DataFrame(
    str_col  = ["red", "blue", "green", "red", "blue"],                        # AbstractString
    sym_col  = [:circle, :square, :triangle, :square, :circle],                # Symbol
    cat_col  = categorical(["small", "medium", "large", "small", "large"]),    # CategoricalValue
    uint_col = UInt32[1, 2, 3, 4, 5],                                          # UInt32
    int_col  = Int[10, 20, 30, 40, 50]                                         # Int
)
```

```@example
dt = DataTreatment(df)
```

```@example
df = DataFrame(
    str_col  = [missing, missing, "green", "red", "blue"],                     # AbstractString
    sym_col  = [missing, :square, :triangle, :square, :circle],                # Symbol
    cat_col  = categorical([missing, "medium", "large", "small", missing]),    # CategoricalValue
    uint_col = Union{Missing, UInt32}[missing, 2, 3, 4, 5],                    # UInt32
    int_col  = Union{Missing, Int}[missing, 20, 30, 40, 50]                    # Int
)
```

```@example
dt = DataTreatment(df)
```

```@example
# ---------------------------------------------------------------------------- #
#                      dataset with only scalar features                       #
# ---------------------------------------------------------------------------- #
df = DataFrame(
    V1 = [1.0, 2.0, 3.0, 4.0],
    V2 = [2.5, 3.5, 4.5, 5.5],
    V3 = [3.2, 4.2, 5.2, 6.2],
    V4 = [4.1, 5.1, 6.1, 7.1],
    V5 = [5.0, 6.0, 7.0, 8.0]
)
```

```@example
dt = DataTreatment(df)
get_X(dt)
```

```@example
eltype(get_X(dt, :scalar)) == Float64
```

```@example
dt = DataTreatment(df; float_type=Float32)
eltype(get_X(dt, :scalar)) == Float32
```

```@example
df = DataFrame(
    V1 = [missing, 2.0, 3.0, 4.0],
    V2 = [2.5, missing, 4.5, 5.5],
    V3 = [3.2, 4.2, missing, 6.2],
    V4 = [4.1, 5.1, 6.1, missing],
    V5 = [5.0, 6.0, 7.0, 8.0]
)
```

```@example
dt = DataTreatment(df) |> get_X
```

```@example
df = DataFrame(
    V1 = [NaN, 2.0, 3.0, 4.0],
    V2 = [2.5, NaN, 4.5, 5.5],
    V3 = [3.2, 4.2, NaN, 6.2],
    V4 = [4.1, 5.1, 6.1, NaN],
    V5 = [5.0, 6.0, 7.0, 8.0]
)
```

```@example
dt = DataTreatment(df) |> get_X
```

```@example
df = DataFrame(
    V1 = [NaN, 2.0, 3.0, missing],
    V2 = [2.5, NaN, 4.5, 5.5],
    V3 = [3.2, 4.2, missing, 6.2],
    V4 = [missing, 5.1, 6.1, NaN],
    V5 = [5.0, 6.0, 7.0, 8.0]
)
```

```@example
dt = DataTreatment(df) |> get_X
```
