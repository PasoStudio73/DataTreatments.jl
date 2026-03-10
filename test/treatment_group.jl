using Test
using DataTreatments

using DataFrames

# Setup test data
df = DataFrame(
    V1 = [1.0, 2.0, 3.0],
    V2 = [4.0, 5.0, 6.0],
    V3 = [7.0, 8.0, 9.0],
    X1 = [10, 20, 30],
    X2 = [40, 50, 60],
    name_col = ["a", "b", "c"]
)

ds_struct = DatasetStructure(df)

@testset "TreatmentGroup" begin

    # ------------------------------------------------------------------ #
    #                          Construction                               #
    # ------------------------------------------------------------------ #
    @testset "Construction" begin
        @testset "Basic construction with no filters" begin
            tg = TreatmentGroup(ds_struct)
            @test length(tg) == ncol(df)
            @test get_dims(tg) == -1
        end

        @testset "Construction from DataFrame" begin
            tg = TreatmentGroup(df)
            @test length(tg) == ncol(df)
            @test get_dims(tg) == -1
        end

        @testset "Construction from Matrix" begin
            tg = TreatmentGroup(Matrix(df), names(df))
            @test length(tg) == ncol(df)
        end

        @testset "Filter by dims" begin
            tg = TreatmentGroup(ds_struct; dims=0)
            @test length(tg) > 0
            @test get_dims(tg) == 0
        end

        @testset "Filter by regex pattern" begin
            tg = TreatmentGroup(ds_struct; name_expr=r"^V")
            @test length(tg) == 3
            @test all(name -> startswith(name, "V"), get_vnames(tg))
        end

        @testset "Filter by function predicate" begin
            tg = TreatmentGroup(ds_struct; name_expr=name -> startswith(name, "X"))
            @test length(tg) == 2
            @test all(name -> startswith(name, "X"), get_vnames(tg))
        end

        @testset "Filter by vector of names" begin
            tg = TreatmentGroup(ds_struct; name_expr=["V1", "X1"])
            @test length(tg) == 2
            @test Set(get_vnames(tg)) == Set(["V1", "X1"])
        end

        @testset "Filter by datatype" begin
            tg = TreatmentGroup(ds_struct; datatype=Float64)
            @test all(name -> startswith(name, "V"), get_vnames(tg))
        end

        @testset "Combined filters" begin
            tg = TreatmentGroup(ds_struct; name_expr=r"^V", datatype=Float64)
            @test length(tg) == 3
            @test all(name -> startswith(name, "V"), get_vnames(tg))
        end

        @testset "Curried constructor" begin
            curried = TreatmentGroup(dims=0)
            @test curried isa Function
            tg = curried(ds_struct)
            @test tg isa TreatmentGroup
            @test get_dims(tg) == 0
        end
    end

    # ------------------------------------------------------------------ #
    #                          Base Methods                               #
    # ------------------------------------------------------------------ #
    @testset "Base Methods" begin
        tg = TreatmentGroup(ds_struct; name_expr=r"^V")

        @testset "length" begin
            @test length(tg) == 3
        end

        @testset "iterate" begin
            idxs = collect(tg)
            @test length(idxs) == 3
            @test all(idx -> idx isa Int, idxs)
            @test idxs == get_idxs(tg)
        end

        @testset "eachindex" begin
            ei = eachindex(tg)
            @test length(collect(ei)) == length(tg)
        end
    end

    # ------------------------------------------------------------------ #
    #                         Getter Methods                              #
    # ------------------------------------------------------------------ #
    @testset "Getter Methods" begin
        tg = TreatmentGroup(ds_struct; name_expr=r"^V")

        @testset "get_idxs - all indices" begin
            idxs = get_idxs(tg)
            @test idxs isa Vector{Int}
            @test length(idxs) == 3
        end

        @testset "get_idxs - single index" begin
            idx = get_idxs(tg, 1)
            @test idx isa Int
            @test idx == get_idxs(tg)[1]
        end

        @testset "get_dims" begin
            @test get_dims(tg) == -1
        end

        @testset "get_vnames - all names" begin
            @test get_vnames(tg) == ["V1", "V2", "V3"]
        end

        @testset "get_vnames - single name" begin
            @test get_vnames(tg, 1) == "V1"
        end

        @testset "get_vnames - subset by indices" begin
            @test get_vnames(tg, [1, 3]) == ["V1", "V3"]
        end

        @testset "get_aggrfunc" begin
            @test get_aggrfunc(tg) isa Base.Callable
        end

        @testset "get_groupby" begin
            @test get_groupby(tg) == (:vname,)
        end

        @testset "custom groupby" begin
            tg_custom = TreatmentGroup(ds_struct; groupby=(:vname, :window))
            @test get_groupby(tg_custom) == (:vname, :window)
        end
    end

    # ------------------------------------------------------------------ #
    #                     Type parameter T                                #
    # ------------------------------------------------------------------ #
    @testset "Type parameter" begin
        @testset "homogeneous Float64 columns" begin
            tg = TreatmentGroup(ds_struct; name_expr=r"^V")
            @test tg isa TreatmentGroup{Float64}
        end

        @testset "homogeneous Int columns" begin
            tg = TreatmentGroup(ds_struct; name_expr=r"^X")
            @test tg isa TreatmentGroup{Int64}
        end

        @testset "mixed types → typejoin" begin
            tg = TreatmentGroup(ds_struct; name_expr=r"^V|^X")
            T = typeof(tg).parameters[1]
            @test Float64 <: T
            @test Int64 <: T
        end

        @testset "empty selection → Any" begin
            tg = TreatmentGroup(ds_struct; name_expr=r"^NONEXISTENT")
            @test tg isa TreatmentGroup{Any}
        end
    end

    # ------------------------------------------------------------------ #
    #                   Multiple Treatment Groups                        #
    # ------------------------------------------------------------------ #
    @testset "Multiple Treatment Groups" begin
        @testset "get_idxs with vector of TreatmentGroups" begin
            tg1 = TreatmentGroup(ds_struct; name_expr=r"^V")
            tg2 = TreatmentGroup(ds_struct; name_expr=r"^X")
            idxs_vec = get_idxs([tg1, tg2])
            @test length(idxs_vec) == 2
            @test all(idx -> idx isa Vector{Int}, idxs_vec)
        end

        @testset "Disjoint partitioning - no overlap" begin
            tg1 = TreatmentGroup(ds_struct; name_expr=r"^V")
            tg2 = TreatmentGroup(ds_struct; name_expr=r"^X")
            idxs_vec = @test_logs get_idxs([tg1, tg2])
            combined = union(idxs_vec...)
            @test length(combined) == length(tg1) + length(tg2)
            @test isdisjoint(idxs_vec[1], idxs_vec[2])
        end

        @testset "Disjoint partitioning - with overlap, later group takes precedence" begin
            tg1 = TreatmentGroup(ds_struct; name_expr=r"^V")
            tg2 = TreatmentGroup(ds_struct; name_expr=r"V|X")
            idxs_vec = @test_logs (:warn,) get_idxs([tg1, tg2])

            # results must be disjoint
            @test isdisjoint(idxs_vec[1], idxs_vec[2])

            # later group (tg2) keeps all its original indices
            @test Set(idxs_vec[2]) == Set(get_idxs(tg2))

            # earlier group (tg1) loses overlapping indices
            @test issubset(Set(idxs_vec[1]), Set(get_idxs(tg1)))
            @test isempty(idxs_vec[1])  # tg2 covers all of V + X, tg1's V cols are taken
        end

        @testset "Three groups with cascading overlap" begin
            tg1 = TreatmentGroup(ds_struct; name_expr=r"^V")
            tg2 = TreatmentGroup(ds_struct; name_expr=r"V|X")
            tg3 = TreatmentGroup(ds_struct)
            idxs_vec = @test_logs (:warn,) get_idxs([tg1, tg2, tg3])

            # all pairwise disjoint
            for i in 1:3, j in i+1:3
                @test isdisjoint(idxs_vec[i], idxs_vec[j])
            end

            # union covers all columns
            @test Set(union(idxs_vec...)) == Set(get_idxs(tg3))
        end
    end

    # ------------------------------------------------------------------ #
    #                          Edge Cases                                 #
    # ------------------------------------------------------------------ #
    @testset "Edge Cases" begin
        @testset "Empty result from filters" begin
            tg = TreatmentGroup(ds_struct; name_expr=r"^NONEXISTENT")
            @test length(tg) == 0
            @test isempty(get_idxs(tg))
            @test isempty(get_vnames(tg))
        end

        @testset "Single column result" begin
            tg = TreatmentGroup(ds_struct; name_expr=["V1"])
            @test length(tg) == 1
            @test get_vnames(tg) == ["V1"]
        end

        @testset "All columns selected" begin
            tg = TreatmentGroup(ds_struct)
            @test length(tg) == ncol(df)
        end
    end

    # ------------------------------------------------------------------ #
    #                          Show Methods                               #
    # ------------------------------------------------------------------ #
    @testset "show methods" begin
        tg = TreatmentGroup(ds_struct; name_expr=r"^V")

        @testset "one-line show" begin
            io = IOBuffer()
            show(io, tg)
            output = String(take!(io))
            @test contains(output, "TreatmentGroup")
            @test contains(output, "3 cols")
            @test contains(output, "dims=all")
        end

        @testset "multi-line show (text/plain)" begin
            io = IOBuffer()
            show(io, MIME"text/plain"(), tg)
            output = String(take!(io))
            @test contains(output, "TreatmentGroup")
            @test contains(output, "3 columns selected")
            @test contains(output, "dims filter:")
            @test contains(output, "selected indices:")
        end

        @testset "show with dims filter" begin
            tg_dims = TreatmentGroup(ds_struct; dims=0)
            io = IOBuffer()
            show(io, tg_dims)
            output = String(take!(io))
            @test contains(output, "dims=0")
        end

        @testset "multi-line show with dims > 0 (aggregation branch)" begin
            # Dataset with 1D vector columns to trigger dims > 0
            df_vectors = DataFrame(
                ts1 = [collect(1.0:5.0), collect(2.0:6.0), collect(3.0:7.0)],
                ts2 = [collect(4.0:8.0), collect(5.0:9.0), collect(6.0:10.0)],
            )
            ds_vec = DatasetStructure(df_vectors)
            tg_vec = TreatmentGroup(ds_vec; dims=1)

            io = IOBuffer()
            show(io, MIME"text/plain"(), tg_vec)
            output = String(take!(io))
            @test contains(output, "TreatmentGroup")
            @test contains(output, "dims filter:")
            @test contains(output, "selected indices:")
            @test contains(output, "aggregation function:")
            @test contains(output, "groupby:")
        end
    end
end