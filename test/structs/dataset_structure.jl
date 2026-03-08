using Test
using DataTreatments

@testset "DatasetStructure" begin
    
    @testset "Constructor" begin
        # Valid construction
        datatype = [Int64, Float64, String]
        valididxs = [[1, 2], [1, 2, 3], [1]]
        missingidxs = [Int[], Int[], [2]]
        nanidxs = [Int[], [3], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        @test length(ds) == 3
        
        # Length mismatch error
        @test_throws DimensionMismatch DatasetStructure(
            [Int64, Float64],  # 2 elements
            [[1, 2], [1, 2, 3], [1]],  # 3 elements
            missingidxs, nanidxs, hasmissing, hasnans
        )
    end
    
    @testset "Size and length methods" begin
        datatype = [Int64, Float64, String]
        valididxs = [[1, 2], [1, 2, 3], [1]]
        missingidxs = [Int[], Int[], [2]]
        nanidxs = [Int[], [3], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        
        @test size(ds) == (3,)
        @test length(ds) == 3
        @test ndims(ds) == 1
    end
    
    @testset "Getter methods - full vectors" begin
        datatype = [Int64, Float64, String]
        valididxs = [[1, 2], [1, 2, 3], [1]]
        missingidxs = [Int[], Int[], [2]]
        nanidxs = [Int[], [3], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        
        @test get_datatype(ds) == datatype
        @test get_valididxs(ds) == valididxs
        @test get_missingidxs(ds) == missingidxs
        @test get_nanidxs(ds) == nanidxs
        @test get_hasmissing(ds) == hasmissing
        @test get_hasnans(ds) == hasnans
    end
    
    @testset "Getter methods - by index" begin
        datatype = [Int64, Float64, String]
        valididxs = [[1, 2], [1, 2, 3], [1]]
        missingidxs = [Int[], Int[], [2]]
        nanidxs = [Int[], [3], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        
        @test get_datatype(ds, 1) == Int64
        @test get_datatype(ds, 2) == Float64
        @test get_datatype(ds, 3) == String
        
        @test get_valididxs(ds, 1) == [1, 2]
        @test get_missingidxs(ds, 3) == [2]
        @test get_nanidxs(ds, 2) == [3]
    end
    
    @testset "Iteration support" begin
        datatype = [Int64, Float64, String]
        valididxs = [[1, 2], [1, 2, 3], [1]]
        missingidxs = [Int[], Int[], [2]]
        nanidxs = [Int[], [3], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        
        types = collect(ds)
        @test types == [Int64, Float64, String]
        
        @test collect(eachindex(ds)) == [1, 2, 3]
    end
    
    @testset "get_structure method" begin
        datatype = [Int64, Float64, Int64, String]
        valididxs = [[1, 2], [1, 2, 3], [1], [1, 2]]
        missingidxs = [Int[], Int[], [2], [1]]
        nanidxs = [Int[], [3], Int[], Int[]]
        hasmissing = [Int[], Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        structure = get_structure(ds)
        
        @test structure.ncols == 4
        @test haskey(structure.type_to_cols, Int64)
        @test haskey(structure.type_to_cols, Float64)
        @test haskey(structure.type_to_cols, String)
        
        @test structure.type_to_cols[Int64] == [1, 3]
        @test structure.type_to_cols[Float64] == [2]
        @test structure.type_to_cols[String] == [4]
        
        @test Set(structure.cols_with_missing) == Set([3, 4])
        @test Set(structure.cols_with_nans) == Set([2])
    end
    
    @testset "show method" begin
        datatype = [Int64, Float64, String]
        valididxs = [[1, 2], [1, 2, 3], [1]]
        missingidxs = [Int[], Int[], [2]]
        nanidxs = [Int[], [3], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        
        # Test that show doesn't error
        io = IOBuffer()
        show(io, ds)
        output = String(take!(io))
        
        @test contains(output, "DatasetStructure(3 columns)")
        @test contains(output, "datatypes by columns:")
        @test contains(output, "missing at:")
        @test contains(output, "NaN at:")
    end
end