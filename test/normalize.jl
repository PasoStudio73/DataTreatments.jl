using Test
using DataTreatments
const DT = DataTreatments

@test ZScore == ZScore
@test ZScore() == (ZScore, nothing)
@test ZScore(dims=2) == (ZScore, 2)
@test ZScore(method=:std) == (ZScore, nothing)
@test ZScore(method=:robust) == (DT.ZScoreRobust, nothing)
@test ZScore(method=:half) == (DT.HalfZScore, nothing)
@test_throws ErrorException ZScore(dims=3)
@test_throws ErrorException ZScore(method=:invalid)

@test MinMax == MinMax
@test MinMax() == (MinMax, nothing)
@test MinMax(dims=2) == (MinMax, 2)
@test MinMax(lower=-1) == (DT.ScaledMinMax, nothing, -1, 1.0)
@test MinMax(upper=10) == (DT.ScaledMinMax, nothing, 0.0, 10)
@test MinMax(lower=-1, upper=10) == (DT.ScaledMinMax, nothing, -1, 10)
@test_throws ErrorException MinMax(dims=3)
@test_throws ErrorException MinMax(lower=10, upper=-1)

@test Scale == Scale
@test Scale() == (Scale, nothing)
@test Scale(dims=2) == (Scale, 2)
@test Scale(method=:std) == (Scale, nothing)
@test Scale(method=:mad) == (DT.ScaleMad, nothing)
@test Scale(method=:first) == (DT.ScaleFirst, nothing)
@test Scale(method=:iqr) == (DT.ScaleIqr, nothing)
@test_throws ErrorException Scale(dims=3)
@test_throws ErrorException Scale(method=:invalid)

@test Sigmoid == Sigmoid
@test Sigmoid() == (Sigmoid, nothing)
@test Sigmoid(dims=2) == (Sigmoid, 2)
@test_throws ErrorException Sigmoid(dims=3)

@test Center == Center
@test Center() == (Center, nothing)
@test Center(dims=2) == (Center, 2)
@test Center(method=:mean) == (Center, nothing)
@test Center(method=:median) == (DT.CenterMedian, nothing)
@test_throws ErrorException Center(dims=3)
@test_throws ErrorException Center(method=:invalid)

@test UnitEnergy == UnitEnergy
@test UnitEnergy() == (UnitEnergy, nothing)
@test UnitEnergy(dims=2) == (UnitEnergy, 2)
@test_throws ErrorException UnitEnergy(dims=3)

@test UnitPower == UnitPower
@test UnitPower() == (UnitPower, nothing)
@test UnitPower(dims=2) == (UnitPower, 2)
@test_throws ErrorException UnitPower(dims=3)

@test PNorm == PNorm
@test PNorm() == (PNorm, nothing, 2.0)
@test PNorm(dims=2) == (PNorm, 2, 2.0)
@test PNorm(p=4) == (PNorm, nothing, 4)
@test_throws ErrorException PNorm(dims=3)
@test_throws ErrorException PNorm(p=-1)

# ---------------------------------------------------------------------------- #
#                                 normalization                                #
# ---------------------------------------------------------------------------- #
X = Float64.([8 1 6; 3 5 7; 4 9 2])

all_elements = normalize(X, MinMax)
@test all_elements == [
    0.875 0.0 0.625;
    0.25 0.5 0.75;
    0.375 1.0 0.125
]

grouby_cols = normalize(X, MinMax; dims=1)
@test grouby_cols == [
    1.0 0.0 0.8;
    0.0 0.5 1.0;
    0.2 1.0 0.0
]

grouby_rows = normalize(X, MinMax; dims=2)
@test isapprox(grouby_rows, [
    1.0 0.0 0.714285714;
    0.0 0.5 1.0;
    0.285714286 1.0 0.0
])

Xmatrix = [Float64.(rand(1:100, 4, 2)) for _ in 1:10, _ in 1:5]

@test_nowarn normalize(Xmatrix, ZScore)
@test_nowarn normalize(Xmatrix, ZScore; dims=1)
@test_nowarn normalize(Xmatrix, ZScore; dims=2)

# ---------------------------------------------------------------------------- #
#                            multi dim normalization                           #
# ---------------------------------------------------------------------------- #
m1 = [1.0 1.0 1.0; 1.0 2.5 1.0; 1.0 1.0 1.0]
m2 = [1.0 1.0 1.0; 1.0 7.5 1.0; 1.0 1.0 1.0]
m3 = [9.0 9.0 9.0; 9.0 2.5 9.0; 9.0 9.0 9.0]
m4 = [9.0 9.0 9.0; 9.0 7.5 9.0; 9.0 9.0 9.0]

M = reshape([m1, m2, m3, m4], 2, 2) # 2x2 matrix of matrices

multidim_norm = normalize(M, MinMax)

# all elements of the matrices were scaled by the same coefficient,
# computed using all values across the matrices.
@test multidim_norm[1,1] ==
    [0.0 0.0 0.0; 0.0 0.1875 0.0; 0.0 0.0 0.0]
@test multidim_norm[1,2] == 
    [1.0 1.0 1.0; 1.0 0.1875 1.0; 1.0 1.0 1.0]
@test multidim_norm[2,1] ==
    [0.0 0.0 0.0; 0.0 0.8125 0.0; 0.0 0.0 0.0]
@test multidim_norm[2,2] ==
    [1.0 1.0 1.0; 1.0 0.8125 1.0; 1.0 1.0 1.0]

# ---------------------------------------------------------------------------- #
#                             tabular normalization                            #
# ---------------------------------------------------------------------------- #
X = Float64.([8 1 6; 3 5 7; 4 9 2])

# test values verified against MATLAB
zscore_norm = normalize(X, ZScore; dims=1)
@test isapprox(zscore_norm, [1.13389 -1.0 0.377964; -0.755929 0.0 0.755929; -0.377964 1.0 -1.13389], atol=1e-5)

zscore_row = normalize(X, ZScore; dims=2)
@test isapprox(zscore_row, [0.83205 -1.1094 0.27735; -1.0 0.0 1.0; -0.27735 1.1094 -0.83205], atol=1e-5)

zscore_robust = normalize(X, ZScore(:robust); dims=1)
@test zscore_robust == [4.0 -1.0 0.0; -1.0 0.0 1.0; 0.0 1.0 -4.0]

@test_nowarn zscore_half = normalize(X, ZScore(:half); dims=2)

@test_nowarn normalize(X, Sigmoid; dims=2)

scale_norm = normalize(X, Scale; dims=1)
@test isapprox(scale_norm, [3.02372 0.25 2.26779; 1.13389 1.25 2.64575; 1.51186 2.25 0.755929], atol=1e-5)

scale_norm = normalize(X, Scale(:mad); dims=1)
@test scale_norm == [8.0 0.25 6.0; 3.0 1.25 7.0; 4.0 2.25 2.0]

scale_norm = normalize(X, Scale(:first); dims=1)
@test isapprox(scale_norm, [1.0 1.0 1.0; 0.375 5.0 1.16667; 0.5 9.0 0.333333], atol=1e-5)

@test_nowarn scale_norm = normalize(X, Scale(:iqr); dims=1)

minmax_norm = normalize(X, MinMax; dims=1)
@test minmax_norm == [1.0 0.0 0.8; 0.0 0.5 1.0; 0.2 1.0 0.0]

minmax_norm = normalize(X, ScaledMinMax; dims=1, lower=-2, upper=4)
@test minmax_norm == [4.0 -2.0 2.8; -2.0 1.0 4.0; -0.8 4.0 -2.0]

center_norm = normalize(X, Center; dims=1)
@test center_norm == [3.0 -4.0 1.0; -2.0 0.0 2.0; -1.0 4.0 -3.0]

center_norm = normalize(X, Center(:median); dims=1)
@test center_norm == [4.0 -4.0 0.0; -1.0 0.0 1.0; 0.0 4.0 -4.0]

@test_nowarn normalize(X, UnitEnergy; dims=1)
@test_nowarn normalize(X, UnitPower; dims=1)

# assolutamente da verificare
@test_nowarn normalize(X, OutlierSuppress; dims=1)

norm_norm = normalize(X, PNorm; dims=1)
@test isapprox(norm_norm, [0.533333 0.0666667 0.4; 0.2 0.333333 0.466667; 0.266667 0.6 0.133333], atol=1e-5)

norm_norm = normalize(X, PNorm(:_2); dims=1)
@test isapprox(norm_norm, [0.847998 0.0966736 0.635999; 0.317999 0.483368 0.741999; 0.423999 0.870063 0.212], atol=1e-6)

norm_norm = normalize(X, PNorm(:max); dims=1)
@test isapprox(norm_norm, [1.0 0.111111 0.857143; 0.375 0.555556 1.0; 0.5 1.0 0.285714], atol=1e-6)

# test against julia package Normalization
X = rand(200,100)

# ---------------------------------------------------------------------------- #
#                              benchmark test                                  #
# ---------------------------------------------------------------------------- #
# test against julia package Normalization
# X = rand(2000,1000)

# test = normalize(X, ZScore; dims=2)
# n = fit(ZScore, X, dims=1)
# norm = Normalization.normalize(X, n)
# @test isapprox(test, norm)

# @btime test = normalize(X, ZScore; dims=1)
# 13.703 ms (6006 allocations: 45.91 MiB)
# 2.178 ms (178 allocations: 15.29 MiB)

# @btime begin
#     n = fit(ZScore, X, dims=1)
#     norm = Normalization.normalize(X, n)   
# end
# 2.245 ms (178 allocations: 15.29 MiB)