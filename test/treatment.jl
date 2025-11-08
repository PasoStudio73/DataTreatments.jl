using Test
using DataTreatments

using DataFrames
using Statistics

@testset "applyfeat - Single Array" begin
    @testset "1D array" begin
        X = rand(100)
        wfunc = splitwindow(nwindows=10)
        intervals = @evalwindow X wfunc
        result = applyfeat(X, intervals)
        
        @test size(result) == (10,)
        @test eltype(result) == Float64
    end
    
    @testset "2D array with default mean" begin
        X = rand(100, 120)
        wfunc = splitwindow(nwindows=3)
        intervals = @evalwindow X wfunc
        result = applyfeat(X, intervals)
        
        @test size(result) == (3, 3)
        @test eltype(result) == Float64
    end
    
    @testset "2D array with custom reducefunc" begin
        X = rand(100, 120)
        wfunc = splitwindow(nwindows=3)
        intervals = @evalwindow X wfunc
        result = applyfeat(X, intervals; reducefunc=maximum)
        
        @test size(result) == (3, 3)
        @test all(result .>= 0) && all(result .<= 1)
    end
end

@testset "aggregate - Flatten to Tabular" begin
    X = rand(100, 120)
    Xmatrix = fill(X, 100, 10)
    wfunc = splitwindow(nwindows=3)
    intervals = @evalwindow X wfunc
    features = (mean, maximum)
    
    result = aggregate(Xmatrix, intervals; features)
    
    @test size(result, 1) == size(Xmatrix, 1)  # Same number of rows
    @test size(result, 2) == size(Xmatrix, 2) * prod(length.(intervals)) * length(features)
    @test eltype(result) == Float64
end

@testset "reducesize - Reduce elements size" begin
    X = rand(100, 120)
    Xmatrix = fill(X, 100, 10)
    wfunc = splitwindow(nwindows=3)
    intervals = @evalwindow X wfunc
    
    result = reducesize(Xmatrix, intervals; reducefunc=Statistics.std)
    
    @test size(result) == size(Xmatrix)
    @test eltype(result) == typeof(first(result))
    @test size(first(result)) == (3, 3)
end

@testset "FeatureId" begin
    @testset "Creation and accessors" begin
        fid = FeatureId(:temperature, mean, 1)
        
        @test get_vname(fid) == :temperature
        @test get_feature(fid) == mean
        @test get_nwin(fid) == 1
    end
    
    @testset "Display single window" begin
        fid = FeatureId(:temperature, mean, 1)
        str = sprint(show, fid)
        @test occursin("mean", str)
        @test occursin("temperature", str)
        @test !occursin("_w", str)
    end
    
    @testset "Display multi-window" begin
        fid = FeatureId(:pressure, maximum, 3)
        str = sprint(show, fid)
        @test occursin("maximum", str)
        @test occursin("pressure", str)
        @test occursin("_w3", str)
    end
end

@testset "DataTreatment - Matrix Input" begin
    Xmatrix = fill(rand(200, 120), 100, 10)
    win = splitwindow(nwindows=4)
    features = (mean, std, maximum)
    
    @testset ":reducesize mode" begin
        dt = DataTreatment(Xmatrix, :reducesize; 
                            vnames=Symbol.("var", 1:10),
                            win=(win,), 
                            reducefunc=mean)
        
        @test size(dt, 1) == 100
        @test get_aggrtype(dt) == :reducesize
        @test length(get_vnames(dt)) == 10
    end
    
    @testset ":aggregate mode" begin
        dt = DataTreatment(Xmatrix, :aggregate;
                            vnames=Symbol.("var", 1:10),
                            win=(win,),
                            features=features)
        
        @test size(dt) == (100, 10 * length(features) * 16)  # 10 vars × 3 features × 16 windows
        @test length(get_featureid(dt)) == size(dt, 2)
        @test get_aggrtype(dt) == :aggregate
        @test length(get_vnames(dt)) == 480
        @test length(get_features(dt)) == length(features)
    end
end