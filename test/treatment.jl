using Test
using DataTreatments
const DT = DataTreatments

using DataFrames
using Statistics

@testset "treatment.jl" begin
    @testset "get_window_ranges" begin
        @testset "1D intervals" begin
            intervals = ([1:3, 4:6],)
            idx = CartesianIndex(1)
            @test DT.get_window_ranges(intervals, idx) == (1:3,)
            idx = CartesianIndex(2)
            @test DT.get_window_ranges(intervals, idx) == (4:6,)
        end

        @testset "2D intervals" begin
            intervals = ([1:2, 3:4], [1:5, 6:10])
            idx = CartesianIndex(1, 1)
            @test DT.get_window_ranges(intervals, idx) == (1:2, 1:5)
            idx = CartesianIndex(2, 2)
            @test DT.get_window_ranges(intervals, idx) == (3:4, 6:10)
            idx = CartesianIndex(1, 2)
            @test DT.get_window_ranges(intervals, idx) == (1:2, 6:10)
        end

        @testset "3D intervals" begin
            intervals = ([1:2, 3:4], [1:3], [1:5, 6:10])
            idx = CartesianIndex(2, 1, 2)
            @test DT.get_window_ranges(intervals, idx) == (3:4, 1:3, 6:10)
        end
    end

    @testset "is_multidim_dataset" begin
        @testset "non-multidimensional data" begin
            # Simple scalar matrix
            X = [1.0 2.0; 3.0 4.0]
            @test DT.is_multidim_dataset(X) == false

            # DataFrame with scalar columns
            df = DataFrame(a=[1.0, 2.0], b=[3.0, 4.0])
            @test DT.is_multidim_dataset(df) == false
        end

        @testset "multidimensional data (array elements)" begin
            # Matrix with array-valued elements
            X = Matrix{Any}(undef, 2, 2)
            X[1,1] = [1.0, 2.0, 3.0]
            X[2,1] = [4.0, 5.0, 6.0]
            X[1,2] = [7.0, 8.0, 9.0]
            X[2,2] = [10.0, 11.0, 12.0]
            Xtyped = Array{Vector{Float64}}(X)
            @test DT.is_multidim_dataset(Xtyped) == true

            # DataFrame with array columns
            df = DataFrame(
                a=[[1.0, 2.0], [3.0, 4.0]],
                b=[[5.0, 6.0], [7.0, 8.0]]
            )
            @test DT.is_multidim_dataset(df) == true
        end

        @testset "mixed columns in DataFrame" begin
            df = DataFrame(
                scalar=[1.0, 2.0],
                vector=[[1.0, 2.0], [3.0, 4.0]]
            )
            @test DT.is_multidim_dataset(df) == true
        end
    end

    @testset "has_uniform_element_size" begin
        @testset "DataFrame" begin
            @testset "empty DataFrame" begin
                df = DataFrame()
                @test DT.has_uniform_element_size(df) == true
            end

            @testset "uniform sizes" begin
                df = DataFrame(
                    a=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    b=[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
                )
                @test DT.has_uniform_element_size(df) == true
            end

            @testset "non-uniform sizes" begin
                df = DataFrame(
                    a=Vector{Vector{Float64}}([[1.0, 2.0], [4.0, 5.0, 6.0]]),
                )
                @test DT.has_uniform_element_size(df) == false
            end

            @testset "with missing values - uniform" begin
                df = DataFrame(
                    a=Vector{Union{Missing,Vector{Float64}}}([missing, [4.0, 5.0, 6.0]]),
                    b=Vector{Union{Missing,Vector{Float64}}}([[7.0, 8.0, 9.0], missing])
                )
                @test DT.has_uniform_element_size(df) == true
            end

            @testset "all missing returns true" begin
                df = DataFrame(
                    a=Union{Missing,Vector{Float64}}[missing, missing],
                )
                @test DT.has_uniform_element_size(df) == true
            end
        end

        @testset "AbstractArray" begin
            @testset "empty array" begin
                X = Vector{Float64}()
                @test DT.has_uniform_element_size(X) == true
            end

            @testset "uniform sizes" begin
                X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
                @test DT.has_uniform_element_size(X) == true
            end

            @testset "non-uniform sizes" begin
                X = Vector{Vector{Float64}}([[1.0, 2.0], [3.0, 4.0, 5.0]])
                @test DT.has_uniform_element_size(X) == false
            end

            @testset "with missing values" begin
                X = Union{Missing,Vector{Float64}}[missing, [1.0, 2.0], [3.0, 4.0]]
                @test DT.has_uniform_element_size(X) == true
            end

            @testset "with NaN floats" begin
                X = [NaN, 1.0, 2.0]
                @test DT.has_uniform_element_size(X) == true
            end

            @testset "all missing returns true" begin
                X = Union{Missing,Float64}[missing, missing]
                @test DT.has_uniform_element_size(X) == true
            end

            @testset "2D arrays uniform" begin
                X = [ones(3, 3), ones(3, 3)]
                @test DT.has_uniform_element_size(X) == true
            end

            @testset "2D arrays non-uniform" begin
                X = Matrix{Float64}[ones(3, 3), ones(2, 4)]
                @test DT.has_uniform_element_size(X) == false
            end
        end
    end

    @testset "safe_feat" begin
        @testset "basic functionality" begin
            v = [1.0, 2.0, 3.0, 4.0, 5.0]
            @test DT.safe_feat(v, mean) ≈ 3.0
            @test DT.safe_feat(v, maximum) ≈ 5.0
            @test DT.safe_feat(v, minimum) ≈ 1.0
            @test DT.safe_feat(v, sum) ≈ 15.0
        end

        @testset "with missing values" begin
            v = Union{Missing,Float64}[1.0, missing, 3.0, missing, 5.0]
            @test DT.safe_feat(v, mean) ≈ 3.0
            @test DT.safe_feat(v, maximum) ≈ 5.0
            @test DT.safe_feat(v, minimum) ≈ 1.0
        end

        @testset "with NaN values" begin
            v = [1.0, NaN, 3.0, NaN, 5.0]
            @test DT.safe_feat(v, mean) ≈ 3.0
            @test DT.safe_feat(v, maximum) ≈ 5.0
            @test DT.safe_feat(v, minimum) ≈ 1.0
        end

        @testset "with both missing and NaN" begin
            v = Union{Missing,Float64}[1.0, missing, NaN, 4.0, missing, NaN, 7.0]
            @test DT.safe_feat(v, mean) ≈ 4.0
            @test DT.safe_feat(v, maximum) ≈ 7.0
            @test DT.safe_feat(v, minimum) ≈ 1.0
        end

        @testset "single element" begin
            v = [42.0]
            @test DT.safe_feat(v, mean) ≈ 42.0
        end

        @testset "integer input" begin
            v = [1, 2, 3]
            @test DT.safe_feat(v, mean) ≈ 2.0
            @test DT.safe_feat(v, sum) == 6
        end
    end

    @testset "aggregate" begin
        @testset "1D vectors with wholewindow" begin
            # 3 observations, 2 features (columns), each element is a 1D vector
            X = Matrix{Union{Missing,Vector{Float64}}}(undef, 3, 2)
            X[1, 1] = [1.0, 2.0, 3.0, 4.0]
            X[2, 1] = [5.0, 6.0, 7.0, 8.0]
            X[3, 1] = [9.0, 10.0, 11.0, 12.0]
            X[1, 2] = [10.0, 20.0]
            X[2, 2] = [30.0, 40.0]
            X[3, 2] = [50.0, 60.0]

            idx = [collect(1:3), collect(1:3)]
            features = (maximum, minimum, mean)
            win = (wholewindow(),)

            Xa, nwindows = aggregate(X, idx, Float64; win=win, features=features)

            # wholewindow -> 1 window per dimension, 3 features => 3 columns per original col
            @test nwindows == [1, 1]
            @test size(Xa) == (3, 6)  # 3 rows, (1*3 + 1*3) = 6 columns

            # Column 1: max of [1,2,3,4] = 4, min = 1, mean = 2.5
            @test Xa[1, 1] ≈ 4.0   # max col1
            @test Xa[1, 2] ≈ 1.0   # min col1
            @test Xa[1, 3] ≈ 2.5   # mean col1

            # Column 2: max of [10,20] = 20, min = 10, mean = 15
            @test Xa[1, 4] ≈ 20.0  # max col2
            @test Xa[1, 5] ≈ 10.0  # min col2
            @test Xa[1, 6] ≈ 15.0  # mean col2
        end

        @testset "with missing observations" begin
            X = Matrix{Union{Missing,Vector{Float64},Float64}}(undef, 3, 1)
            X[1, 1] = [1.0, 2.0, 3.0]
            X[2, 1] = missing
            X[3, 1] = [7.0, 8.0, 9.0]

            idx = [[1, 3]]  # row 2 is missing
            features = (mean,)
            win = (wholewindow(),)

            Xa, nwindows = aggregate(X, idx, Float64; win=win, features=features)

            @test nwindows == [1]
            @test size(Xa) == (3, 1)
            @test Xa[1, 1] ≈ 2.0
            @test ismissing(Xa[2, 1])
            @test Xa[3, 1] ≈ 8.0
        end

        @testset "aggregate closure constructor" begin
            f = aggregate(
                win=(wholewindow(),),
                features=(mean, maximum)
            )
            @test f isa Function
        end
    end

    @testset "reducesize" begin
        @testset "1D vectors with wholewindow" begin
            X = Matrix{Union{Missing,Vector{Float64}}}(undef, 2, 2)
            X[1, 1] = [1.0, 2.0, 3.0, 4.0]
            X[2, 1] = [5.0, 6.0, 7.0, 8.0]
            X[1, 2] = [10.0, 20.0]
            X[2, 2] = [30.0, 40.0]

            idx = [collect(1:2), collect(1:2)]
            win = (wholewindow(),)

            Xr = reducesize(X, idx, Float64; win=win, reducefunc=mean)

            @test size(Xr) == size(X)
            # wholewindow reduces each vector to a single-element array
            @test Xr[1, 1] ≈ [2.5]
            @test Xr[2, 1] ≈ [6.5]
            @test Xr[1, 2] ≈ [15.0]
            @test Xr[2, 2] ≈ [35.0]
        end

        @testset "with missing observations" begin
            X = Matrix{Union{Missing,Vector{Float64},Float64}}(undef, 3, 1)
            X[1, 1] = [1.0, 2.0, 3.0]
            X[2, 1] = missing
            X[3, 1] = [7.0, 8.0, 9.0]

            idx = [[1, 3]]
            win = (wholewindow(),)

            Xr = reducesize(X, idx, Float64; win=win, reducefunc=mean)

            @test size(Xr) == (3, 1)
            @test Xr[1, 1] ≈ [2.0]
            @test ismissing(Xr[2, 1])
            @test Xr[3, 1] ≈ [8.0]
        end

        @testset "reducesize closure constructor" begin
            f = reducesize(
                win=(wholewindow(),),
                reducefunc=mean
            )
            @test f isa Function
        end

        @testset "reduction preserves structure" begin
            # 2D array elements
            X = Matrix{Union{Missing,Matrix{Float64}}}(undef, 2, 1)
            X[1, 1] = [1.0 2.0; 3.0 4.0]
            X[2, 1] = [5.0 6.0; 7.0 8.0]

            idx = [collect(1:2)]
            win = (wholewindow(), wholewindow())

            Xr = reducesize(X, idx, Float64; win=win, reducefunc=mean)

            @test size(Xr) == (2, 1)
            # wholewindow on both dims -> single element output
            @test Xr[1, 1] ≈ [2.5;;]  # mean of [1,2,3,4]
            @test Xr[2, 1] ≈ [6.5;;]  # mean of [5,6,7,8]
        end
    end

    @testset "aggregate with split windows" begin
        @testset "splitwindow into 2 parts" begin
            X = Matrix{Union{Missing,Vector{Float64}}}(undef, 2, 1)
            X[1, 1] = [1.0, 2.0, 3.0, 4.0]
            X[2, 1] = [5.0, 6.0, 7.0, 8.0]

            idx = [collect(1:2)]
            features = (mean,)
            win = (splitwindow(nwindows=2,),)

            Xa, nwindows = aggregate(X, idx, Float64; win=win, features=features)

            @test nwindows == [2]
            @test size(Xa) == (2, 2)  # 2 windows * 1 feature = 2 columns
            # First window: [1,2] mean=1.5; Second window: [3,4] mean=3.5
            @test Xa[1, 1] ≈ 1.5
            @test Xa[1, 2] ≈ 3.5
            @test Xa[2, 1] ≈ 5.5
            @test Xa[2, 2] ≈ 7.5
        end

        @testset "splitwindow with multiple features" begin
            X = Matrix{Union{Missing,Vector{Float64}}}(undef, 1, 1)
            X[1, 1] = [1.0, 2.0, 3.0, 4.0]

            idx = [collect(1:1)]
            features = (maximum, minimum)
            win = (splitwindow(nwindows=2,),)

            Xa, nwindows = aggregate(X, idx, Float64; win=win, features=features)

            @test nwindows == [2]
            @test size(Xa) == (1, 4)  # 2 windows * 2 features = 4 columns
            # max window1: 2.0, max window2: 4.0, min window1: 1.0, min window2: 3.0
            @test Xa[1, 1] ≈ 2.0  # max [1,2]
            @test Xa[1, 2] ≈ 4.0  # max [3,4]
            @test Xa[1, 3] ≈ 1.0  # min [1,2]
            @test Xa[1, 4] ≈ 3.0  # min [3,4]
        end
    end

    @testset "reducesize with split windows" begin
        @testset "splitwindow into 2 parts" begin
            X = Matrix{Union{Missing,Vector{Float64}}}(undef, 2, 1)
            X[1, 1] = [1.0, 2.0, 3.0, 4.0]
            X[2, 1] = [5.0, 6.0, 7.0, 8.0]

            idx = [collect(1:2)]
            win = (splitwindow(nwindows=2,),)

            Xr = reducesize(X, idx, Float64; win=win, reducefunc=mean)

            @test size(Xr) == (2, 1)
            @test length(Xr[1, 1]) == 2
            @test Xr[1, 1][1] ≈ 1.5   # mean of [1,2]
            @test Xr[1, 1][2] ≈ 3.5   # mean of [3,4]
            @test Xr[2, 1][1] ≈ 5.5   # mean of [5,6]
            @test Xr[2, 1][2] ≈ 7.5   # mean of [7,8]
        end
    end

    @testset "edge cases" begin
        @testset "single element vectors" begin
            X = Matrix{Union{Missing,Vector{Float64}}}(undef, 1, 1)
            X[1, 1] = [42.0]

            idx = [[1]]
            Xa, nwindows = aggregate(X, idx, Float64;
                win=(wholewindow(),), features=(mean,))
            @test Xa[1, 1] ≈ 42.0
        end

        @testset "Float32 output type" begin
            X = Matrix{Union{Missing,Vector{Float64}}}(undef, 1, 1)
            X[1, 1] = [1.0, 2.0, 3.0]

            idx = [[1]]
            Xa, _ = aggregate(X, idx, Float32;
                win=(wholewindow(),), features=(mean,))
            @test eltype(Xa) == Union{Missing,Float32}
            @test Xa[1, 1] isa Float32
        end

        @testset "multiple columns with different dimensions" begin
            X = Matrix{Union{Missing,Vector{Float64}}}(undef, 2, 2)
            X[1, 1] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            X[2, 1] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
            X[1, 2] = [10.0, 20.0]
            X[2, 2] = [30.0, 40.0]

            idx = [collect(1:2), collect(1:2)]
            features = (mean,)
            win = (wholewindow(),)

            Xa, nwindows = aggregate(X, idx, Float64; win=win, features=features)
            @test nwindows == [1, 1]
            @test size(Xa, 1) == 2
        end
    end

end