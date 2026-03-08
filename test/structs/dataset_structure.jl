using Test
using DataTreatments

@testset "DatasetStructure" begin
    
    @testset "Constructor" begin
        # Valid construction
        datatype = [Int64, Float64, String]
        dims = [0, 0, 0]
        valididxs = [[1, 2], [1, 2, 3], [1]]
        missingidxs = [Int[], Int[], [2]]
        nanidxs = [Int[], [3], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        @test length(ds) == 3
        
        # Length mismatch error
        @test_throws DimensionMismatch DatasetStructure(
            [Int64, Float64],  # 2 elements
            [0, 0, 0],  # 3 elements
            [[1, 2], [1, 2, 3], [1]],  # 3 elements
            missingidxs, nanidxs, hasmissing, hasnans
        )
    end
    
    @testset "Size and length methods" begin
        datatype = [Int64, Float64, String]
        dims = [0, 0, 0]
        valididxs = [[1, 2], [1, 2, 3], [1]]
        missingidxs = [Int[], Int[], [2]]
        nanidxs = [Int[], [3], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        
        @test size(ds) == (3,)
        @test length(ds) == 3
        @test ndims(ds) == 1
    end
    
    @testset "Getter methods - full vectors" begin
        datatype = [Int64, Float64, String]
        dims = [0, 0, 0]
        valididxs = [[1, 2], [1, 2, 3], [1]]
        missingidxs = [Int[], Int[], [2]]
        nanidxs = [Int[], [3], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        
        @test get_datatype(ds) == datatype
        @test get_dims(ds) == dims
        @test get_valididxs(ds) == valididxs
        @test get_missingidxs(ds) == missingidxs
        @test get_nanidxs(ds) == nanidxs
        @test get_hasmissing(ds) == hasmissing
        @test get_hasnans(ds) == hasnans
    end
    
    @testset "Getter methods - by index" begin
        datatype = [Int64, Float64, String]
        dims = [0, 1, 0]
        valididxs = [[1, 2], [1, 2, 3], [1]]
        missingidxs = [Int[], Int[], [2]]
        nanidxs = [Int[], [3], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        
        @test get_datatype(ds, 1) == Int64
        @test get_datatype(ds, 2) == Float64
        @test get_datatype(ds, 3) == String
        
        @test get_dims(ds, 1) == 0
        @test get_dims(ds, 2) == 1
        @test get_dims(ds, 3) == 0
        
        @test get_valididxs(ds, 1) == [1, 2]
        @test get_missingidxs(ds, 3) == [2]
        @test get_nanidxs(ds, 2) == [3]
    end
    
    @testset "Iteration support" begin
        datatype = [Int64, Float64, String]
        dims = [0, 1, 0]
        valididxs = [[1, 2], [1, 2, 3], [1]]
        missingidxs = [Int[], Int[], [2]]
        nanidxs = [Int[], [3], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        
        types = collect(ds)
        @test types == [Int64, Float64, String]
        
        @test collect(eachindex(ds)) == [1, 2, 3]
    end
    
    @testset "get_structure method" begin
        datatype = [Int64, Float64, Int64, String]
        dims = [0, 0, 1, 0]
        valididxs = [[1, 2], [1, 2, 3], [1], [1, 2]]
        missingidxs = [Int[], Int[], [2], [1]]
        nanidxs = [Int[], [3], Int[], Int[]]
        hasmissing = [Int[], Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
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
        dims = [0, 0, 0]
        valididxs = [[1, 2], [1, 2, 3], [1]]
        missingidxs = [Int[], Int[], [2]]
        nanidxs = [Int[], [3], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[]]
        
        ds = DatasetStructure(datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        
        # Test that show doesn't error
        io = IOBuffer()
        show(io, ds)
        output = String(take!(io))
        
        @test contains(output, "DatasetStructure(3 columns)")
        @test contains(output, "datatypes by columns:")
        @test contains(output, "missing at:")
        @test contains(output, "NaN at:")
    end
    
    @testset "Dims handling - vector/array dimensions" begin
        # Scalar columns have dims=0
        # Vector columns have dims=1
        # Matrix columns have dims=2
        datatype = [Float64, Vector{Float64}, Matrix{Float64}]
        dims = [0, 1, 2]
        valididxs = [[1, 2, 3], [1, 2], [1, 2, 3]]
        missingidxs = [Int[], [3], Int[]]
        nanidxs = [Int[], Int[], Int[]]
        hasmissing = [Int[], Int[], Int[]]
        hasnans = [Int[], [2], [1]]
        
        ds = DatasetStructure(datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        
        @test get_dims(ds, 1) == 0
        @test get_dims(ds, 2) == 1
        @test get_dims(ds, 3) == 2
        @test get_dims(ds) == [0, 1, 2]
    end
    
    @testset "Complex scenario with mixed data types and issues" begin
        # Simulate a complex dataset with multiple types, missing, and NaN
        datatype = [String, Int64, Int64, Float64, Float64, Vector{Float64}, Matrix{Float64}]
        dims = [0, 0, 0, 0, 0, 1, 2]
        valididxs = [[1,2,3,4,5], [1,2,3,4,5], [1,2,4,5], [1,2,3,4,5], [1,2,3,4,5], [2,3,4,5], [1,2,3,4,5]]
        missingidxs = [Int[], Int[], [3], Int[], Int[], [1], Int[]]
        nanidxs = [Int[], Int[], Int[], [5], [4], Int[], [2]]
        hasmissing = [Int[], Int[], Int[], Int[], Int[], Int[], Int[]]
        hasnans = [Int[], Int[], Int[], Int[], Int[], [3], [4]]
        
        ds = DatasetStructure(datatype, dims, valididxs, missingidxs, nanidxs, hasmissing, hasnans)
        structure = get_structure(ds)
        
        @test structure.ncols == 7
        @test structure.type_to_cols[Float64] == [4, 5]
        @test structure.type_to_cols[Int64] == [2, 3]
        
        # Columns with issues: 3 (missing), 4 (NaN), 5 (NaN), 6 (hasnans), 7 (hasnans)
        @test Set(structure.cols_with_missing) == Set([3, 6])
        @test Set(structure.cols_with_nans) == Set([4, 5, 6, 7])
    end
end