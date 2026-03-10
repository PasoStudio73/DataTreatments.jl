using Test
using DataTreatments

using DataFrames

@testset "DatasetStructure" begin
    @testset "Constructor - Matrix" begin
        dataset = Matrix{Any}(undef, 5, 3)
        dataset[:, 1] = [1, 2, missing, 4, 5]
        dataset[:, 2] = [1.0, NaN, 3.0, missing, 5.0]
        dataset[:, 3] = [collect(1.0:3.0), collect(2.0:4.0), collect(3.0:5.0), missing, NaN]
        col_names = ["a", "b", "c"]

        ds = DatasetStructure(dataset, col_names)

        @test length(ds) == 3
        @test size(ds) == (3,)
        @test get_vnames(ds) == col_names
    end

    @testset "Constructor - DataFrame" begin
        df = DataFrame(
            a = [1, 2, missing, 4],
            b = [1.0, NaN, 3.0, 4.0],
            c = [collect(1.0:3.0), collect(2.0:4.0), missing, NaN],
        )

        ds = DatasetStructure(df)

        @test length(ds) == 3
        @test get_vnames(ds) == ["a", "b", "c"]
    end

    @testset "Constructor - default vnames" begin
        dataset = Matrix{Any}(undef, 3, 2)
        dataset[:, 1] = [1, 2, 3]
        dataset[:, 2] = [4.0, 5.0, 6.0]

        ds = DatasetStructure(dataset)

        @test get_vnames(ds) == ["V1", "V2"]
    end

    @testset "Size and length methods" begin
        dataset = Matrix{Any}(undef, 4, 3)
        dataset[:, 1] = [1, 2, 3, 4]
        dataset[:, 2] = [1.0, 2.0, 3.0, 4.0]
        dataset[:, 3] = ["a", "b", "c", "d"]

        ds = DatasetStructure(dataset, ["col1", "col2", "col3"])

        @test size(ds) == (3,)
        @test length(ds) == 3
    end

    @testset "Scalar columns - missing and NaN" begin
        dataset = Matrix{Any}(undef, 5, 3)
        dataset[:, 1] = [1, 2, missing, 4, 5]           # Int with 1 missing
        dataset[:, 2] = [1.0, NaN, 3.0, missing, 5.0]   # Float with 1 NaN, 1 missing
        dataset[:, 3] = ["a", "b", "c", "d", "e"]       # String, no issues

        ds = DatasetStructure(dataset, ["a", "b", "c"])

        # col 1: Int with one missing at index 3
        @test get_missingidxs(ds, 1) == [3]
        @test get_nanidxs(ds, 1) == Int[]
        @test get_valididxs(ds, 1) == [1, 2, 4, 5]
        @test get_dims(ds, 1) == 0

        # col 2: Float with NaN at 2, missing at 4
        @test get_missingidxs(ds, 2) == [4]
        @test get_nanidxs(ds, 2) == [2]
        @test get_valididxs(ds, 2) == [1, 3, 5]
        @test get_dims(ds, 2) == 0

        # col 3: all valid
        @test get_missingidxs(ds, 3) == Int[]
        @test get_nanidxs(ds, 3) == Int[]
        @test get_valididxs(ds, 3) == [1, 2, 3, 4, 5]
        @test get_dims(ds, 3) == 0
    end

    @testset "Array columns - dims, hasmissing, hasnans" begin
        dataset = Matrix{Any}(undef, 5, 2)
        dataset[:, 1] = [
            collect(1.0:3.0),        # valid vector
            [1.0, missing, 3.0],     # vector with internal missing
            [1.0, NaN, 3.0],         # vector with internal NaN
            missing,                 # top-level missing
            NaN,                     # top-level NaN
        ]
        dataset[:, 2] = [
            [1.0 2.0; 3.0 4.0],      # valid matrix
            [missing 2.0; 3.0 4.0],  # matrix with internal missing
            [1.0 NaN; 3.0 4.0],      # matrix with internal NaN
            [1.0 2.0; 3.0 4.0],      # valid matrix
            missing,                 # top-level missing
        ]

        ds = DatasetStructure(dataset, ["vectors", "matrices"])

        # col 1: vectors
        @test get_dims(ds, 1) == 1
        @test get_valididxs(ds, 1) == [1, 2, 3]
        @test get_missingidxs(ds, 1) == [4]
        @test get_nanidxs(ds, 1) == [5]
        @test get_hasmissing(ds, 1) == [2]
        @test get_hasnans(ds, 1) == [3]

        # col 2: matrices
        @test get_dims(ds, 2) == 2
        @test get_valididxs(ds, 2) == [1, 2, 3, 4]
        @test get_missingidxs(ds, 2) == [5]
        @test get_nanidxs(ds, 2) == Int[]
        @test get_hasmissing(ds, 2) == [2]
        @test get_hasnans(ds, 2) == [3]
    end

    @testset "Datatype inference" begin
        dataset = Matrix{Any}(undef, 4, 3)
        dataset[:, 1] = [1, 2, 3, 4]                     # all Int64
        dataset[:, 2] = [1.0, 2.0, 3.0, 4.0]             # all Float64
        dataset[:, 3] = [1, 2.0, 3, 4.0]                 # Int64 + Float64 → Real

        ds = DatasetStructure(dataset, ["ints", "floats", "mixed"])

        @test get_datatype(ds, 1) == Int64
        @test get_datatype(ds, 2) == Float64
        @test get_datatype(ds, 3) <: Real
    end

    @testset "Getter methods - full vectors" begin
        dataset = Matrix{Any}(undef, 3, 2)
        dataset[:, 1] = [1, missing, 3]
        dataset[:, 2] = [1.0, NaN, 3.0]

        ds = DatasetStructure(dataset, ["a", "b"])

        @test get_vnames(ds) == ["a", "b"]
        @test length(get_datatype(ds)) == 2
        @test length(get_dims(ds)) == 2
        @test length(get_valididxs(ds)) == 2
        @test length(get_missingidxs(ds)) == 2
        @test length(get_nanidxs(ds)) == 2
        @test length(get_hasmissing(ds)) == 2
        @test length(get_hasnans(ds)) == 2
    end

    @testset "Getter methods - by index" begin
        dataset = Matrix{Any}(undef, 4, 3)
        dataset[:, 1] = [1, 2, missing, 4]
        dataset[:, 2] = [1.0, NaN, 3.0, 4.0]
        dataset[:, 3] = ["a", "b", "c", "d"]

        ds = DatasetStructure(dataset, ["col1", "col2", "col3"])

        @test get_vnames(ds, 1) == "col1"
        @test get_vnames(ds, 2) == "col2"
        @test get_vnames(ds, 3) == "col3"
        @test get_vnames(ds, [1, 3]) == ["col1", "col3"]

        @test get_datatype(ds, 1) == Int64
        @test get_datatype(ds, 2) == Float64
        @test get_datatype(ds, 3) == String
        @test get_datatype(ds, [1, 2]) == [Int64, Float64]

        @test get_dims(ds, 1) == 0
        @test get_dims(ds, [1, 2, 3]) == [0, 0, 0]

        @test get_missingidxs(ds, 1) == [3]
        @test get_missingidxs(ds, [1, 2]) == [get_missingidxs(ds, 1), get_missingidxs(ds, 2)]

        @test get_nanidxs(ds, 2) == [2]
        @test get_nanidxs(ds, [1, 3]) == [get_nanidxs(ds, 1), get_nanidxs(ds, 3)]

        @test get_valididxs(ds, 3) == [1, 2, 3, 4]
        @test get_valididxs(ds, [1, 3]) == [get_valididxs(ds, 1), get_valididxs(ds, 3)]

        @test get_hasmissing(ds, [1, 2, 3]) == [get_hasmissing(ds, 1), get_hasmissing(ds, 2), get_hasmissing(ds, 3)]

        @test get_hasnans(ds, [1, 2, 3]) == [get_hasnans(ds, 1), get_hasnans(ds, 2), get_hasnans(ds, 3)]
    end

    @testset "All clean dataset" begin
        dataset = Matrix{Any}(undef, 3, 2)
        dataset[:, 1] = [1, 2, 3]
        dataset[:, 2] = [4.0, 5.0, 6.0]

        ds = DatasetStructure(dataset, ["a", "b"])

        @test get_missingidxs(ds, 1) == Int[]
        @test get_missingidxs(ds, 2) == Int[]
        @test get_nanidxs(ds, 1) == Int[]
        @test get_nanidxs(ds, 2) == Int[]
        @test get_hasmissing(ds, 1) == Int[]
        @test get_hasnans(ds, 2) == Int[]
        @test get_valididxs(ds, 1) == [1, 2, 3]
        @test get_valididxs(ds, 2) == [1, 2, 3]
    end

    @testset "show methods" begin
        dataset = Matrix{Any}(undef, 3, 3)
        dataset[:, 1] = [1, 2, missing]
        dataset[:, 2] = [1.0, NaN, 3.0]
        dataset[:, 3] = ["a", "b", "c"]

        ds = DatasetStructure(dataset, ["col1", "col2", "col3"])

        # Test one-line show
        io = IOBuffer()
        show(io, ds)
        output = String(take!(io))
        @test contains(output, "DatasetStructure(3 cols)")

        # Test multi-line show (text/plain)
        io = IOBuffer()
        show(io, MIME"text/plain"(), ds)
        output = String(take!(io))
        @test contains(output, "DatasetStructure(3 columns)")
        @test contains(output, "datatypes by columns:")
        @test contains(output, "missing at:")
        @test contains(output, "NaN at:")
    end

    @testset "show method - clean dataset" begin
        dataset = Matrix{Any}(undef, 3, 2)
        dataset[:, 1] = [1, 2, 3]
        dataset[:, 2] = [4.0, 5.0, 6.0]

        ds = DatasetStructure(dataset, ["a", "b"])

        io = IOBuffer()
        show(io, MIME"text/plain"(), ds)
        output = String(take!(io))
        @test !contains(output, "missing at:")
        @test !contains(output, "NaN at:")
    end
end