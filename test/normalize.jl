using Test
using DataTreatments
const DT = DataTreatments

using Normalization

# ---------------------------------------------------------------------------- #
#                             tabular normalization                            #
# ---------------------------------------------------------------------------- #
a = [8 1 6; 3 5 7; 4 9 2]

# test values verified against MATLAB
zscore_norm = tabular_norm(a, zscore())
@test isapprox(zscore_norm, [1.13389 -1.0 0.377964; -0.755929 0.0 0.755929; -0.377964 1.0 -1.13389], atol=1e-5)

zscore_row = tabular_norm(a, zscore(); dim=:row)
@test isapprox(zscore_row, [0.83205 -1.1094 0.27735; -1.0 0.0 1.0; -0.27735 1.1094 -0.83205], atol=1e-5)

zscore_robust = tabular_norm(a, zscore(method=:robust))
@test zscore_robust == [4.0 -1.0 0.0; -1.0 0.0 1.0; 0.0 1.0 -4.0]

zscore_half = tabular_norm(a, zscore(method=:half))

@test_throws ArgumentError tabular_norm(a, zscore(); dim=:invalid)
@test_throws ArgumentError tabular_norm(a, zscore(method=:invalid))

@test_nowarn tabular_norm(a, sigmoid())

norm_norm = tabular_norm(a, pnorm())
@test isapprox(norm_norm, [0.847998 0.0966736 0.635999; 0.317999 0.483368 0.741999; 0.423999 0.870063 0.212], atol=1e-6)

norm_norm = tabular_norm(a, pnorm(p=4))
@test isapprox(norm_norm, [0.980428 0.108608 0.768635; 0.36766 0.543042 0.896741; 0.490214 0.977475 0.256212], atol=1e-5)

norm_norm = tabular_norm(a, pnorm(p=Inf))
@test isapprox(norm_norm, [1.0 0.111111 0.857143; 0.375 0.555556 1.0; 0.5 1.0 0.285714], atol=1e-6)

scale_norm = tabular_norm(a, scale(factor=:std))
@test isapprox(scale_norm, [3.02372 0.25 2.26779; 1.13389 1.25 2.64575; 1.51186 2.25 0.755929], atol=1e-5)

scale_norm = tabular_norm(a, scale(factor=:mad))
@test scale_norm == [8.0 0.25 6.0; 3.0 1.25 7.0; 4.0 2.25 2.0]

scale_norm = tabular_norm(a, scale(factor=:first))
@test isapprox(scale_norm, [1.0 1.0 1.0; 0.375 5.0 1.16667; 0.5 9.0 0.333333], atol=1e-5)

scale_norm = tabular_norm(a, scale(factor=:iqr))

minmax_norm = tabular_norm(a, DT.minmax())
@test minmax_norm == [1.0 0.0 0.8; 0.0 0.5 1.0; 0.2 1.0 0.0]

minmax_norm = tabular_norm(a, DT.minmax(lower=-2, upper=4))
@test minmax_norm == [4.0 -2.0 2.8; -2.0 1.0 4.0; -0.8 4.0 -2.0]

center_norm = tabular_norm(a, center())
@test center_norm == [3.0 -4.0 1.0; -2.0 0.0 2.0; -1.0 4.0 -3.0]

center_norm = tabular_norm(a, center(method=:median))
@test center_norm == [4.0 -4.0 0.0; -1.0 0.0 1.0; 0.0 4.0 -4.0]

@test_nowarn tabular_norm(a, unitpower())

@test_nowarn tabular_norm(a, outliersuppress())
@test_nowarn tabular_norm(a, outliersuppress(thr=3))

# test against julia package Normalization
X = rand(200,100)

test = tabular_norm(X, zscore())
n = fit(ZScore, X, dims=1)
norm = normalize(X, n)
@test isapprox(test, norm)

test = tabular_norm(X, zscore(method=:half))
n = fit(HalfZScore, X, dims=1)
norm = normalize(X, n)
@test isapprox(test, norm)

test = tabular_norm(X, sigmoid())
n = fit(Sigmoid, X, dims=1)
norm = normalize(X, n)
@test isapprox(test, norm)

test = tabular_norm(X, pnorm())
n = fit(UnitEnergy, X, dims=1)
norm = normalize(X, n)
@test isapprox(test, norm)

test = tabular_norm(X, DT.minmax())
n = fit(MinMax, X, dims=1)
norm = normalize(X, n)
@test isapprox(test, norm)

test = tabular_norm(X, center())
n = fit(Center, X, dims=1)
norm = normalize(X, n)
@test isapprox(test, norm)

test = tabular_norm(X, unitpower())
n = fit(UnitPower, X, dims=1)
norm = normalize(X, n)
@test isapprox(test, norm)

test = tabular_norm(X, outliersuppress(;thr=5))
n = fit(OutlierSuppress, X, dims=1)
norm = normalize(X, n)
@test isapprox(test, norm)

# ---------------------------------------------------------------------------- #
#                        single element normalization                          #
# ---------------------------------------------------------------------------- #
X = rand(100,75, 2)

@test_nowarn element_norm(X, zscore())
@test_nowarn element_norm(X, sigmoid())
@test_nowarn element_norm(X, pnorm())
@test_nowarn element_norm(X, scale())
@test_nowarn element_norm(X, DT.minmax())
@test_nowarn element_norm(X, center())
@test_nowarn element_norm(X, unitpower())
@test_nowarn element_norm(X, outliersuppress())

# test against julia package Normalization
X = rand(200,100)

test = element_norm(X, zscore())
n = fit(ZScore, X)
norm = normalize(X, n)
@test isapprox(test, norm)

test = element_norm(X, zscore(method=:half))
n = fit(HalfZScore, X)
norm = normalize(X, n)
@test isapprox(test, norm)

test = element_norm(X, sigmoid())
n = fit(Sigmoid, X)
norm = normalize(X, n)
@test isapprox(test, norm)

test = element_norm(X, pnorm())
n = fit(UnitEnergy, X)
norm = normalize(X, n)
@test isapprox(test, norm)

test = element_norm(X, DT.minmax())
n = fit(MinMax, X)
norm = normalize(X, n)
@test isapprox(test, norm)

test = element_norm(X, center())
n = fit(Center, X)
norm = normalize(X, n)
@test isapprox(test, norm)

test = element_norm(X, unitpower())
n = fit(UnitPower, X)
norm = normalize(X, n)
@test isapprox(test, norm)

test = element_norm(X, outliersuppress(;thr=5))
n = fit(OutlierSuppress, X)
norm = normalize(X, n)
@test isapprox(test, norm)

# ---------------------------------------------------------------------------- #
#                    n-dimensional dataset normalization                       #
# ---------------------------------------------------------------------------- #
X = [rand(200, 100) .* 1000 for _ in 1:100, _ in 1:100]

@test_nowarn ds_norm(X, zscore())
@test_nowarn ds_norm(X, sigmoid())
@test_nowarn ds_norm(X, pnorm())
@test_nowarn ds_norm(X, scale())
@test_nowarn ds_norm(X, DT.minmax())
@test_nowarn ds_norm(X, center())
@test_nowarn ds_norm(X, unitpower())
@test_nowarn ds_norm(X, outliersuppress())

function test_ds_norm(X, norm_func, NormType)
    test = ds_norm(X, norm_func)
    # compute normalization the way ds_norm does (per column)
    col1_data = collect(Iterators.flatten(X[:, 1]))
    n = fit(NormType, reshape(col1_data, :, 1); dims=nothing)
    norm = normalize(X[1,1], n)
    
    @test isapprox(test[1,1], norm)
end

# Run all tests
X = fill(rand(20, 10) .* 10, 10, 100)

test_ds_norm(X, zscore(), ZScore)
test_ds_norm(X, zscore(method=:half), HalfZScore)
test_ds_norm(X, sigmoid(), Sigmoid)
test_ds_norm(X, pnorm(), UnitEnergy)
test_ds_norm(X, DT.minmax(), MinMax)
test_ds_norm(X, center(), Center)
test_ds_norm(X, unitpower(), UnitPower)
test_ds_norm(X, outliersuppress(;thr=5), OutlierSuppress)

