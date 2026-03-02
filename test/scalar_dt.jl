using Test
using DataTreatments
const DT = DataTreatments

using DataFrames
using CategoricalArrays

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

dt = DataTreatment(df)
get_X(dt)

eltype(get_X(dt, :scalar)) == Float64

dt = DataTreatment(df; float_type=Float32)
eltype(get_X(dt, :scalar)) == Float32

df = DataFrame(
    V1 = [missing, 2.0, 3.0, 4.0],
    V2 = [2.5, missing, 4.5, 5.5],
    V3 = [3.2, 4.2, missing, 6.2],
    V4 = [4.1, 5.1, 6.1, missing],
    V5 = [5.0, 6.0, 7.0, 8.0]
)

dt = DataTreatment(df) |> get_X

df = DataFrame(
    V1 = [NaN, 2.0, 3.0, 4.0],
    V2 = [2.5, NaN, 4.5, 5.5],
    V3 = [3.2, 4.2, NaN, 6.2],
    V4 = [4.1, 5.1, 6.1, NaN],
    V5 = [5.0, 6.0, 7.0, 8.0]
)

dt = DataTreatment(df) |> get_X

df = DataFrame(
    V1 = [NaN, 2.0, 3.0, missing],
    V2 = [2.5, NaN, 4.5, 5.5],
    V3 = [3.2, 4.2, missing, 6.2],
    V4 = [missing, 5.1, 6.1, NaN],
    V5 = [5.0, 6.0, 7.0, 8.0]
)

dt = DataTreatment(df) |> get_X