using Test
using DataTreatments
const DT = DataTreatments

using DataFrames
using CategoricalArrays
using Random
using Statistics

# ---------------------------------------------------------------------------- #
#                               helper functions                               #
# ---------------------------------------------------------------------------- #
function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

# helper: find datasets of a given type in a Vector{AbstractDataset}
find_discrete(ds::Vector{<:DT.AbstractDataset}) = filter(d -> d isa DiscreteDataset, ds)
find_continuous(ds::Vector{<:DT.AbstractDataset}) = filter(d -> d isa ContinuousDataset, ds)
find_multidim(ds::Vector{<:DT.AbstractDataset}) = filter(d -> d isa MultidimDataset, ds)

# ---------------------------------------------------------------------------- #
#                     dataset with only discrete features                      #
# ---------------------------------------------------------------------------- #
@testset "Discrete-only datasets" begin
    @testset "Clean discrete dataset" begin
        df = DataFrame(
            str_col  = ["red", "blue", "green", "red", "blue"],
            sym_col  = [:circle, :square, :triangle, :square, :circle],
            cat_col  = categorical(["small", "medium", "large", "small", "large"]),
            uint_col = UInt32[1, 2, 3, 4, 5],
            int_col  = Int[10, 20, 30, 40, 50]
        )

        dt = DataTreatment(df)
        datasets = get_datasets(dt)

        @test !isempty(datasets)

        dds = find_discrete(datasets)
        cds = find_continuous(datasets)
        mds = find_multidim(datasets)

        @test length(dds) >= 1
        @test isempty(cds)
        @test isempty(mds)

        dd = first(dds)
        @test dd isa DiscreteDataset
        @test size(dd, 1) == 5
        @test size(dd, 2) == 5
        @test length(dd) == 5
        @test all(f -> f isa DiscreteFeat, dd)

        # variable names
        vnames = get_vnames(dd)
        @test "str_col" in vnames
        @test "sym_col" in vnames
        @test "cat_col" in vnames
        @test "uint_col" in vnames
        @test "int_col" in vnames
    end

    @testset "Discrete dataset with missing values" begin
        df = DataFrame(
            str_col  = [missing, missing, "green", "red", "blue"],
            sym_col  = [missing, :square, :triangle, :square, :circle],
            cat_col  = categorical([missing, "medium", "large", "small", missing]),
            uint_col = Union{Missing, UInt32}[missing, 2, 3, 4, 5],
            int_col  = Union{Missing, Int}[missing, 20, 30, 40, 50]
        )

        dt = DataTreatment(df)
        datasets = get_datasets(dt)

        dds = find_discrete(datasets)
        @test length(dds) >= 1

        dd = first(dds)
        @test size(dd, 1) == 5
        @test size(dd, 2) == 5
    end
end

# ---------------------------------------------------------------------------- #
#                      dataset with only scalar features                       #
# ---------------------------------------------------------------------------- #
@testset "Scalar-only datasets" begin
    @testset "Basic scalar dataset (Float64)" begin
        df = DataFrame(
            V1 = [1.0, 2.0, 3.0, 4.0],
            V2 = [2.5, 3.5, 4.5, 5.5],
            V3 = [3.2, 4.2, 5.2, 6.2],
            V4 = [4.1, 5.1, 6.1, 7.1],
            V5 = [5.0, 6.0, 7.0, 8.0]
        )

        dt = DataTreatment(df)
        datasets = get_datasets(dt)

        dds = find_discrete(datasets)
        cds = find_continuous(datasets)
        mds = find_multidim(datasets)

        @test isempty(dds)
        @test length(cds) >= 1
        @test isempty(mds)

        cd = first(cds)
        @test cd isa ContinuousDataset{Float64}
        @test size(cd) == (4, 5)
        @test length(cd) == 5
        @test all(f -> f isa ContinuousFeat{Float64}, cd)

        vnames = get_vnames(cd)
        @test vnames == ["V1", "V2", "V3", "V4", "V5"]
    end

    @testset "Scalar dataset with Float32" begin
        df = DataFrame(
            V1 = [1.0, 2.0, 3.0, 4.0],
            V2 = [2.5, 3.5, 4.5, 5.5],
            V3 = [3.2, 4.2, 5.2, 6.2],
        )

        dt = DataTreatment(df; float_type=Float32)
        datasets = get_datasets(dt)

        cds = find_continuous(datasets)
        @test length(cds) >= 1

        cd = first(cds)
        @test cd isa ContinuousDataset{Float32}
        @test size(cd) == (4, 3)
    end

    @testset "Scalar dataset with missing values" begin
        df = DataFrame(
            V1 = [missing, 2.0, 3.0, 4.0],
            V2 = [2.5, missing, 4.5, 5.5],
            V3 = [3.2, 4.2, missing, 6.2],
            V4 = [4.1, 5.1, 6.1, missing],
            V5 = [5.0, 6.0, 7.0, 8.0]
        )

        dt = DataTreatment(df)
        datasets = get_datasets(dt)
        cds = find_continuous(datasets)
        @test length(cds) >= 1

        cd = first(cds)
        @test size(cd) == (4, 5)
    end

    @testset "Scalar dataset with NaN values" begin
        df = DataFrame(
            V1 = [NaN, 2.0, 3.0, 4.0],
            V2 = [2.5, NaN, 4.5, 5.5],
            V3 = [3.2, 4.2, NaN, 6.2],
            V4 = [4.1, 5.1, 6.1, NaN],
            V5 = [5.0, 6.0, 7.0, 8.0]
        )

        dt = DataTreatment(df)
        datasets = get_datasets(dt)
        cds = find_continuous(datasets)
        cd = first(cds)

        @test size(cd) == (4, 5)
        # NaN preserved
        @test isnan(get_dataset(cd)[1, 1])
        @test isnan(get_dataset(cd)[2, 2])
    end

    @testset "Scalar dataset with both NaN and missing" begin
        df = DataFrame(
            V1 = [NaN, 2.0, 3.0, missing],
            V2 = [2.5, NaN, missing, 5.5],
            V3 = [NaN, 4.2, missing, 6.2],
            V4 = [missing, 5.1, 6.1, NaN],
            V5 = [5.0, 6.0, 7.0, 8.0]
        )

        dt = DataTreatment(df)
        datasets = get_datasets(dt)
        cds = find_continuous(datasets)
        cd = first(cds)

        @test size(cd) == (4, 5)
    end
end

# ---------------------------------------------------------------------------- #
#            dataset with only 1D multidimensional features                    #
# ---------------------------------------------------------------------------- #
@testset "1D Multidimensional datasets" begin
    test_configs = [
        (label="default",                  kwargs=(;)),
        (label="Float32",                  kwargs=(; float_type=Float32)),
        (label="reducesize",               kwargs=(; ),),
        (label="reducesize+Float32",       kwargs=(; float_type=Float32)),
        (label="adaptivewindow",           kwargs=(;)),
        (label="adaptivewindow+Float32",   kwargs=(; float_type=Float32)),
    ]

    test_dfs = Dict(
        :clean => DataFrame(
            ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
            ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), collect(5.0:0.5:8.5)],
            ts3 = [collect(1.0:1.2:7.0), collect(2.0:1.2:8.0), collect(0.5:1.2:6.5), collect(1.5:1.2:7.5), collect(3.0:1.2:9.0)],
            ts4 = [collect(6.0:-0.8:1.0), collect(7.0:-0.8:2.0), collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), collect(9.0:-0.8:4.0)]
        ),
        :missing => DataFrame(
            ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
            ts2 = [missing, collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), collect(5.0:0.5:8.5)],
            ts3 = [collect(1.0:1.2:7.0), collect(2.0:1.2:8.0), missing, collect(1.5:1.2:7.5), collect(3.0:1.2:9.0)],
            ts4 = [collect(6.0:-0.8:1.0), missing, collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), missing]
        ),
        :nan => DataFrame(
            ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
            ts2 = [NaN, collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), collect(5.0:0.5:8.5)],
            ts3 = [collect(1.0:1.2:7.0), collect(2.0:1.2:8.0), NaN, collect(1.5:1.2:7.5), collect(3.0:1.2:9.0)],
            ts4 = [collect(6.0:-0.8:1.0), NaN, collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), NaN]
        ),
        :mixed => DataFrame(
            ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
            ts2 = [NaN, collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), missing, collect(5.0:0.5:8.5)],
            ts3 = [missing, collect(2.0:1.2:8.0), NaN, collect(1.5:1.2:7.5), missing],
            ts4 = [missing, NaN, collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), NaN]
        )
    )

    for (df_name, df) in test_dfs
        @testset "DataFrame: $df_name" begin
            @test is_multidim_dataset(df) == true
            @test has_uniform_element_size(df) == false

            @testset "Default aggregate" begin
                dt = DataTreatment(df)
                datasets = get_datasets(dt)

                @test !isempty(datasets)
                mds = find_multidim(datasets)
                @test length(mds) >= 1

                for md in mds
                    @test size(md, 1) == 5
                    @test size(md, 2) > 0
                    @test all(f -> f isa AggregateFeat, md)
                end
            end

            @testset "Float32 aggregate" begin
                dt = DataTreatment(df; float_type=Float32)
                datasets = get_datasets(dt)
                mds = find_multidim(datasets)
                @test length(mds) >= 1

                for md in mds
                    @test md isa MultidimDataset{Float32}
                    @test size(md, 1) == 5
                end
            end

            @testset "Reducesize" begin
                dt = DataTreatment(df,
                    TreatmentGroup(aggrfunc=reducesize()))
                datasets = get_datasets(dt)
                mds = find_multidim(datasets)
                @test length(mds) >= 1

                for md in mds
                    @test all(f -> f isa ReduceFeat, md)
                    @test size(md, 1) == 5
                end
            end
            @testset "Reducesize + Float32" begin
                dt = DataTreatment(df,
                    TreatmentGroup(aggrfunc=reducesize());
                    float_type=Float32)
                datasets = get_datasets(dt)
                mds = find_multidim(datasets)
                @test length(mds) >= 1

                for md in mds
                    @test md isa MultidimDataset{<:AbstractArray{Float32}}
                    @test all(f -> f isa ReduceFeat, md)
                end
            end

            @testset "Custom features (mean only)" begin
                dt = DataTreatment(df,
                    TreatmentGroup(aggrfunc=aggregate(features=(mean,))))
                datasets = get_datasets(dt)
                mds = find_multidim(datasets)
                @test length(mds) >= 1

                for md in mds
                    @test all(f -> f isa AggregateFeat, md)
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#          dataset with only 2D multidimensional features (images)             #
# ---------------------------------------------------------------------------- #
@testset "2D Multidimensional datasets (images)" begin
    test_dfs = Dict(
        :clean => DataFrame(
            img1 = [create_image(i) for i in 1:5],
            img2 = [create_image(i+10) for i in 1:5],
            img3 = [create_image(i+20) for i in 1:5],
            img4 = [create_image(i+30) for i in 1:5]
        ),
        :missing => DataFrame(
            img1 = [create_image(i) for i in 1:5],
            img2 = [i == 1 ? missing : create_image(i+10) for i in 1:5],
            img3 = [create_image(i+20) for i in 1:5],
            img4 = [i == 3 ? missing : create_image(i+30) for i in 1:5]
        ),
        :nan => DataFrame(
            img1 = [create_image(i) for i in 1:5],
            img2 = [i == 1 ? NaN : create_image(i+10) for i in 1:5],
            img3 = [create_image(i+20) for i in 1:5],
            img4 = [i == 3 ? NaN : create_image(i+30) for i in 1:5]
        ),
        :mixed => DataFrame(
            img1 = [create_image(i) for i in 1:5],
            img2 = [i == 1 ? NaN : (i == 4 ? missing : create_image(i+10)) for i in 1:5],
            img3 = [i == 3 ? missing : create_image(i+20) for i in 1:5],
            img4 = [i == 2 ? NaN : (i == 5 ? missing : create_image(i+30)) for i in 1:5]
        )
    )

    for (df_name, df) in test_dfs
        @testset "DataFrame: $df_name" begin
            @test is_multidim_dataset(df) == true
            @test has_uniform_element_size(df) == true

            @testset "Default aggregate" begin
                dt = DataTreatment(df)
                datasets = get_datasets(dt)
                mds = find_multidim(datasets)
                @test length(mds) >= 1

                for md in mds
                    @test size(md, 1) == 5
                    @test size(md, 2) > 0
                    @test all(f -> f isa AggregateFeat, md)
                    # all 2D → dims should be 2
                    @test all(d -> d == 2, get_dims(md))
                end
            end

            @testset "Reducesize" begin
                dt = DataTreatment(df,
                    TreatmentGroup(aggrfunc=reducesize()))
                datasets = get_datasets(dt)
                mds = find_multidim(datasets)
                @test length(mds) >= 1

                for md in mds
                    @test all(f -> f isa ReduceFeat, md)
                    @test size(md, 1) == 5
                end
            end

            @testset "Float32" begin
                dt = DataTreatment(df; float_type=Float32)
                datasets = get_datasets(dt)
                mds = find_multidim(datasets)
                for md in mds
                    @test md isa MultidimDataset{Float32} || md isa MultidimDataset{<:AbstractArray{Float32}}
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#                        non-homogeneous dataset                               #
# ---------------------------------------------------------------------------- #
@testset "Non-homogeneous dataset (discrete + scalar + 1D + 2D)" begin
    df_clean = DataFrame(
        str_col  = ["red", "blue", "green", "red", "blue"],
        sym_col  = [:circle, :square, :triangle, :square, :circle],
        cat_col  = categorical(["small", "medium", "large", "small", "large"]),
        uint_col = UInt32[1, 2, 3, 4, 5],
        int_col  = Int[10, 20, 30, 40, 50],
        V1 = [1.0, 2.0, 3.0, 4.0, 5.6],
        V2 = [2.5, 3.5, 4.5, 5.5, 7.8],
        V3 = [3.2, 4.2, 5.2, 6.2, 2.4],
        V4 = [4.1, 5.1, 6.1, 7.1, 5.5],
        V5 = [5.0, 6.0, 7.0, 8.0, 1.8],
        ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
        ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), collect(5.0:0.5:8.5)],
        ts3 = [collect(1.0:1.2:7.0), collect(2.0:1.2:8.0), collect(0.5:1.2:6.5), collect(1.5:1.2:7.5), collect(3.0:1.2:9.0)],
        ts4 = [collect(6.0:-0.8:1.0), collect(7.0:-0.8:2.0), collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), collect(9.0:-0.8:4.0)],
        img1 = [create_image(i) for i in 1:5],
        img2 = [create_image(i+10) for i in 1:5],
        img3 = [create_image(i+20) for i in 1:5],
        img4 = [create_image(i+30) for i in 1:5]
    )

    @testset "Detection" begin
        @test is_multidim_dataset(df_clean) == true
    end

    @testset "Default aggregate - all three dataset types present" begin
        dt = DataTreatment(df_clean)
        datasets = get_datasets(dt)

        dds = find_discrete(datasets)
        cds = find_continuous(datasets)
        mds = find_multidim(datasets)

        @test length(dds) >= 1
        @test length(cds) >= 1
        @test length(mds) >= 1

        # discrete: 5 columns (str, sym, cat, uint, int)
        total_discrete_cols = sum(length(d) for d in dds)
        @test total_discrete_cols == 5

        # continuous: 5 columns (V1..V5)
        total_continuous_cols = sum(length(d) for d in cds)
        @test total_continuous_cols == 5

        # multidim: split by dims → should have at least 2 MultidimDatasets (1D and 2D)
        unique_dims_found = unique(reduce(vcat, [get_dims(md) for md in mds]))
        @test 1 in unique_dims_found
        @test 2 in unique_dims_found

        # each should have correct number of rows
        for md in mds
            @test size(md, 1) == 5
        end
    end

    @testset "Dimension splitting" begin
        dt = DataTreatment(df_clean)
        datasets = get_datasets(dt)
        mds = find_multidim(datasets)

        # each MultidimDataset should have homogeneous dims
        for md in mds
            dims = get_dims(md)
            @test length(unique(dims)) == 1
        end
    end

    @testset "Float32 conversion" begin
        dt = DataTreatment(df_clean; float_type=Float32)
        datasets = get_datasets(dt)

        cds = find_continuous(datasets)
        for cd in cds
            @test cd isa ContinuousDataset{Float32}
        end

        mds = find_multidim(datasets)
        for md in mds
            @test md isa MultidimDataset{Float32} || md isa MultidimDataset{<:AbstractArray{Float32}}
        end
    end

    @testset "Reducesize mode" begin
        dt = DataTreatment(df_clean,
            TreatmentGroup(aggrfunc=reducesize()))
        datasets = get_datasets(dt)
        mds = find_multidim(datasets)
        @test length(mds) >= 1

        for md in mds
            @test all(f -> f isa ReduceFeat, md)
            @test size(md, 1) == 5
        end
    end

    @testset "Custom features (mean only)" begin
        dt = DataTreatment(df_clean,
            TreatmentGroup(aggrfunc=aggregate(features=(mean,))))
        datasets = get_datasets(dt)
        mds = find_multidim(datasets)

        for md in mds
            @test all(f -> f isa AggregateFeat, md)
        end
    end

    @testset "With missing values" begin
        df_miss = DataFrame(
            str_col  = [missing, "blue", "green", "red", "blue"],
            sym_col  = [:circle, :square, :triangle, :square, missing],
            cat_col  = categorical(["small", "medium", missing, "small", "large"]),
            uint_col = UInt32[1, 2, 3, 4, 5],
            int_col  = Int[10, 20, 30, 40, 50],
            V1 = [missing, 2.0, 3.0, 4.0, 5.6],
            V2 = [2.5, 3.5, 4.5, 5.5, missing],
            V3 = [3.2, 4.2, 5.2, 6.2, 2.4],
            V4 = [4.1, missing, missing, 7.1, 5.5],
            V5 = [5.0, 6.0, 7.0, 8.0, 1.8],
            ts1 = [missing, collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
            ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), missing],
            ts3 = [collect(1.0:1.2:7.0), missing, missing, collect(1.5:1.2:7.5), collect(3.0:1.2:9.0)],
            ts4 = [collect(6.0:-0.8:1.0), collect(7.0:-0.8:2.0), collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), collect(9.0:-0.8:4.0)],
            img1 = [create_image(i) for i in 1:5],
            img2 = [i == 1 ? missing : create_image(i+10) for i in 1:5],
            img3 = [create_image(i+20) for i in 1:5],
            img4 = [i == 3 ? missing : create_image(i+30) for i in 1:5]
        )

        dt = DataTreatment(df_miss)
        datasets = get_datasets(dt)

        @test !isempty(datasets)
        dds = find_discrete(datasets)
        cds = find_continuous(datasets)
        mds = find_multidim(datasets)

        @test length(dds) >= 1
        @test length(cds) >= 1
        @test length(mds) >= 1
    end

    @testset "With NaN and missing" begin
        df_mix = DataFrame(
            str_col  = [missing, "blue", "green", "red", "blue"],
            sym_col  = [:circle, :square, :triangle, :square, missing],
            cat_col  = categorical(["small", "medium", missing, "small", "large"]),
            uint_col = UInt32[1, 2, 3, 4, 5],
            int_col  = Int[10, 20, 30, 40, 50],
            V1 = [NaN, missing, 3.0, 4.0, 5.6],
            V2 = [2.5, missing, 4.5, 5.5, NaN],
            V3 = [3.2, 4.2, 5.2, missing, 2.4],
            V4 = [4.1, NaN, NaN, 7.1, 5.5],
            V5 = [5.0, 6.0, 7.0, 8.0, 1.8],
            ts1 = [NaN, collect(2.0:7.0), missing, collect(4.0:9.0), collect(5.0:10.0)],
            ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), NaN],
            ts3 = [collect(1.0:1.2:7.0), NaN, NaN, missing, collect(3.0:1.2:9.0)],
            ts4 = [collect(6.0:-0.8:1.0), missing, collect(5.0:-0.8:0.0), collect(8.0:-0.8:3.0), collect(9.0:-0.8:4.0)],
            img1 = [create_image(i) for i in 1:5],
            img2 = [i == 1 ? NaN : create_image(i+10) for i in 1:5],
            img3 = [create_image(i+20) for i in 1:5],
            img4 = [i == 3 ? NaN : create_image(i+30) for i in 1:5]
        )

        dt = DataTreatment(df_mix)
        datasets = get_datasets(dt)

        @test !isempty(datasets)
        dds = find_discrete(datasets)
        cds = find_continuous(datasets)
        mds = find_multidim(datasets)

        @test length(dds) >= 1
        @test length(cds) >= 1
        @test length(mds) >= 1
    end
end

# ---------------------------------------------------------------------------- #
#            get_treatments_datasets vs get_leftover_datasets                  #
# ---------------------------------------------------------------------------- #
@testset "Treatment vs Leftover partitioning" begin
    df = DataFrame(
        str_col  = ["red", "blue", "green", "red", "blue"],
        V1 = [1.0, 2.0, 3.0, 4.0, 5.6],
        V2 = [2.5, 3.5, 4.5, 5.5, 7.8],
        ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
        ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), collect(5.0:0.5:8.5)],
        img1 = [create_image(i) for i in 1:5],
        img2 = [create_image(i+10) for i in 1:5],
    )

    @testset "Default (single treatment group covers everything)" begin
        dt = DataTreatment(df)

        treat_ds = get_treatments_datasets(dt)
        left_ds = get_leftover_datasets(dt)
        all_ds = get_datasets(dt)

        @test !isempty(treat_ds)
        @test length(all_ds) == length(treat_ds) + length(left_ds)
    end

    @testset "get_datasets returns all column types" begin
        dt = DataTreatment(df)
        all_ds = get_datasets(dt)

        dds = find_discrete(all_ds)
        cds = find_continuous(all_ds)
        mds = find_multidim(all_ds)

        total_cols = sum(length(d) for d in dds; init=0) +
                     sum(length(d) for d in cds; init=0) +
                     sum(length(d) for d in mds; init=0)

        # should cover all 7 original columns (possibly with aggregate expansion)
        @test total_cols >= 7
    end
end

# ---------------------------------------------------------------------------- #
#                     getindex on output datasets                              #
# ---------------------------------------------------------------------------- #
@testset "AbstractDataset getindex" begin
    df = DataFrame(
        V1 = [1.0, 2.0, 3.0, 4.0, 5.0],
        V2 = [2.5, 3.5, 4.5, 5.5, 6.5],
        V3 = [3.0, 4.0, 5.0, 6.0, 7.0],
    )

    dt = DataTreatment(df)
    datasets = get_datasets(dt)
    cds = find_continuous(datasets)
    cd = first(cds)

    @testset "Single index" begin
        cd1 = cd[1]
        @test cd1 isa ContinuousDataset
        @test length(cd1) == 1
        @test size(cd1) == (5, 1)
        @test get_info(cd1, 1) === get_info(cd, 1)
    end

    @testset "Vector index" begin
        cd12 = cd[[1, 2]]
        @test cd12 isa ContinuousDataset
        @test length(cd12) == 2
        @test size(cd12, 2) == 2
    end

    @testset "Range index" begin
        cd_range = cd[1:3]
        @test cd_range isa ContinuousDataset
        @test length(cd_range) == 3
    end
end

@testset "MultidimDataset getindex" begin
    df = DataFrame(
        ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
        ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), collect(5.0:0.5:8.5)],
    )

    dt = DataTreatment(df)
    datasets = get_datasets(dt)
    mds = find_multidim(datasets)
    md = first(mds)

    @testset "Single index" begin
        md1 = md[1]
        @test md1 isa MultidimDataset
        @test length(md1) == 1
        @test size(md1, 1) == 5
    end

    @testset "Range index" begin
        md_range = md[1:2]
        @test md_range isa MultidimDataset
        @test length(md_range) == 2
    end
end

@testset "DiscreteDataset getindex" begin
    df = DataFrame(
        str_col = ["red", "blue", "green", "red", "blue"],
        sym_col = [:circle, :square, :triangle, :square, :circle],
    )

    dt = DataTreatment(df)
    datasets = get_datasets(dt)
    dds = find_discrete(datasets)
    dd = first(dds)

    @testset "Single index" begin
        dd1 = dd[1]
        @test dd1 isa DiscreteDataset
        @test length(dd1) == 1
        @test size(dd1) == (5, 1)
    end

    @testset "Range index" begin
        dd_range = dd[1:2]
        @test dd_range isa DiscreteDataset
        @test length(dd_range) == 2
    end
end

# ---------------------------------------------------------------------------- #
#                     _split_md_by_dims                                        #
# ---------------------------------------------------------------------------- #
@testset "_split_md_by_dims" begin
    df = DataFrame(
        ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
        ts2 = [collect(2.0:0.5:5.5), collect(1.0:0.5:4.5), collect(3.0:0.5:6.5), collect(4.0:0.5:7.5), collect(5.0:0.5:8.5)],
        img1 = [create_image(i) for i in 1:5],
        img2 = [create_image(i+10) for i in 1:5],
    )

    dt = DataTreatment(df)
    datasets = get_datasets(dt)
    mds = find_multidim(datasets)

    # should have been split into 1D and 2D
    @test length(mds) >= 2

    dims_per_md = [unique(get_dims(md)) for md in mds]
    # each MultidimDataset should contain features of a single dimensionality
    for dims in dims_per_md
        @test length(dims) == 1
    end

    all_dims = reduce(vcat, dims_per_md)
    @test 1 in all_dims
    @test 2 in all_dims
end

# ---------------------------------------------------------------------------- #
#              Computed values validation                                       #
# ---------------------------------------------------------------------------- #
@testset "Computed values validation" begin
    df = DataFrame(
        ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
        V1  = [1.0, 2.0, 3.0, 4.0, 5.0],
    )

    @testset "Scalar features are preserved" begin
        dt = DataTreatment(df)
        datasets = get_datasets(dt)
        cds = find_continuous(datasets)
        @test length(cds) >= 1

        cd = first(cds)
        mat = get_dataset(cd)
        @test mat[1, 1] ≈ 1.0
        @test mat[2, 1] ≈ 2.0
        @test mat[5, 1] ≈ 5.0
    end

    @testset "Aggregate mean correctness" begin
        dt = DataTreatment(df,
            TreatmentGroup(aggrfunc=aggregate(features=(mean,))))
        datasets = get_datasets(dt)
        mds = find_multidim(datasets)
        @test length(mds) >= 1

        md = first(mds)
        mat = get_dataset(md)

        for rowidx in 1:5
            ts1_values = df.ts1[rowidx]
            expected_mean = mean(ts1_values)
            # find the column for ts1 mean
            ts1_cols = [i for i in eachindex(md.info)
                        if get_vname(md.info[i]) == "ts1"]
            @test !isempty(ts1_cols)
            @test isapprox(mat[rowidx, first(ts1_cols)], expected_mean, atol=1e-10)
        end
    end

    @testset "Multiple features (mean, maximum, minimum)" begin
        dt = DataTreatment(df,
            TreatmentGroup(aggrfunc=aggregate(features=(mean, maximum, minimum))))
        datasets = get_datasets(dt)
        mds = find_multidim(datasets)
        md = first(mds)
        mat = get_dataset(md)

        for rowidx in 1:5
            ts1_values = df.ts1[rowidx]
            ts1_cols = [i for i in eachindex(md.info) if get_vname(md.info[i]) == "ts1"]
            ts1_vals = [mat[rowidx, c] for c in ts1_cols]

            @test any(v -> isapprox(v, mean(ts1_values), atol=1e-10), ts1_vals)
            @test any(v -> isapprox(v, maximum(ts1_values), atol=1e-10), ts1_vals)
            @test any(v -> isapprox(v, minimum(ts1_values), atol=1e-10), ts1_vals)
        end
    end

    @testset "Float32 vs Float64 numerical consistency" begin
        dt_f64 = DataTreatment(df,
            TreatmentGroup(aggrfunc=aggregate(features=(mean,))))
        dt_f32 = DataTreatment(df,
            TreatmentGroup(aggrfunc=aggregate(features=(mean,)));
            float_type=Float32)

        mds_f64 = find_multidim(get_datasets(dt_f64))
        mds_f32 = find_multidim(get_datasets(dt_f32))

        mat_f64 = get_dataset(first(mds_f64))
        mat_f32 = get_dataset(first(mds_f32))

        @test size(mat_f64) == size(mat_f32)
        for i in eachindex(mat_f64)
            if !ismissing(mat_f64[i]) && !ismissing(mat_f32[i])
                @test isapprox(mat_f32[i], mat_f64[i], atol=1e-4)
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#                  Non-homogeneous element sizes (mixed 1D+2D)                 #
# ---------------------------------------------------------------------------- #
@testset "Non-homogeneous element sizes" begin
    df = DataFrame(
        ts1 = [collect(1.0:6.0), collect(2.0:6.0), collect(3.0:9.0), collect(4.0:11.0), collect(5.0:18.0)],
        ts2 = [collect(2.0:0.5:4.5), collect(1.0:0.5:8.5), collect(3.0:0.5:6.5), collect(4.0:0.5:13.5), collect(5.0:0.5:5.5)],
        img1 = [create_image(i; n=6) for i in 1:5],
        img2 = [create_image(i+10; n=7) for i in 1:5],
        img3 = [create_image(i+20; n=5) for i in 1:5],
    )

    @test is_multidim_dataset(df) == true

    @testset "Default aggregate" begin
        dt = DataTreatment(df)
        datasets = get_datasets(dt)
        mds = find_multidim(datasets)

        @test length(mds) >= 2  # split by dims: 1D and 2D

        for md in mds
            @test size(md, 1) == 5
            dims = get_dims(md)
            @test length(unique(dims)) == 1  # homogeneous within each
        end
    end

    @testset "Reducesize" begin
        dt = DataTreatment(df,
            TreatmentGroup(aggrfunc=reducesize()))
        datasets = get_datasets(dt)
        mds = find_multidim(datasets)
        @test length(mds) >= 2

        for md in mds
            @test all(f -> f isa ReduceFeat, md)
        end
    end

    @testset "With missing" begin
        df_miss = DataFrame(
            ts1 = [missing, collect(2.0:6.0), collect(3.0:9.0), collect(4.0:11.0), collect(5.0:18.0)],
            ts2 = [collect(2.0:0.5:4.5), collect(1.0:0.5:8.5), missing, collect(4.0:0.5:13.5), collect(5.0:0.5:5.5)],
            img1 = [create_image(i; n=6) for i in 1:5],
            img2 = [i == 1 ? missing : create_image(i+10; n=7) for i in 1:5],
        )

        dt = DataTreatment(df_miss)
        datasets = get_datasets(dt)
        @test !isempty(datasets)
    end

    @testset "With NaN and missing" begin
        df_mix = DataFrame(
            ts1 = [NaN, missing, collect(3.0:9.0), collect(4.0:11.0), collect(5.0:18.0)],
            ts2 = [collect(2.0:0.5:4.5), missing, NaN, collect(4.0:0.5:13.5), collect(5.0:0.5:5.5)],
            img1 = [create_image(i; n=6) for i in 1:5],
            img2 = [i == 1 ? NaN : create_image(i+10; n=7) for i in 1:5],
        )

        dt = DataTreatment(df_mix)
        datasets = get_datasets(dt)
        @test !isempty(datasets)
    end
end

# ---------------------------------------------------------------------------- #
#                         Display and IO                                       #
# ---------------------------------------------------------------------------- #
@testset "Display and IO" begin
    df = DataFrame(
        str_col = ["x", "y", "z", "x", "y"],
        V1 = [1.0, 2.0, 3.0, 4.0, 5.0],
        ts1 = [collect(1.0:6.0), collect(2.0:7.0), collect(3.0:8.0), collect(4.0:9.0), collect(5.0:10.0)],
    )

    dt = DataTreatment(df)

    @testset "DataTreatment show" begin
        io = IOBuffer()
        show(io, dt)
        output = String(take!(io))
        @test !isempty(output)
        @test contains(output, "DataTreatment")
    end

    @testset "DataTreatment show MIME text/plain" begin
        io = IOBuffer()
        show(io, MIME"text/plain"(), dt)
        output = String(take!(io))
        @test !isempty(output)
    end

    @testset "AbstractDataset show methods" begin
        datasets = get_datasets(dt)

        for ds in datasets
            io = IOBuffer()
            show(io, ds)
            output = String(take!(io))
            @test !isempty(output)

            io = IOBuffer()
            show(io, MIME"text/plain"(), ds)
            output = String(take!(io))
            @test !isempty(output)
        end
    end
end

# ---------------------------------------------------------------------------- #
#                            Base methods                                      #
# ---------------------------------------------------------------------------- #
@testset "DataTreatment Base methods" begin
    df = DataFrame(
        col1 = ["a", "b", "c"],
        col2 = [1.0, 2.0, 3.0],
        col3 = [collect(1.0:5.0), collect(2.0:6.0), collect(3.0:7.0)],
    )

    dt = DataTreatment(df)

    @test size(dt) == (3, 3)
    @test size(dt)[1] == 3
    @test size(dt)[2] == 3
    @test length(dt) == 3
    @test eachindex(dt) == Base.OneTo(3)

    # iterate yields column views
    collected = collect(dt)
    @test length(collected) == 3

    # getter methods
    @test get_nrows(dt) == 3
    @test get_ncols(dt) == 3

    @testset "get_t_groups" begin
        tgroups = get_t_groups(dt)
        @test tgroups isa Vector{TreatmentGroup}
        @test length(tgroups) >= 1

        # single index access
        tg1 = get_t_groups(dt, 1)
        @test tg1 isa TreatmentGroup
        @test tg1 === tgroups[1]
    end

    @testset "get_t_groups with multiple treatment groups" begin
        dt2 = DataTreatment(df,
            TreatmentGroup(dims=0),
            TreatmentGroup(dims=1),
        )

        tgroups = get_t_groups(dt2)
        @test length(tgroups) == 2

        @test get_t_groups(dt2, 1) === tgroups[1]
        @test get_t_groups(dt2, 2) === tgroups[2]

        # out of bounds
        @test_throws BoundsError get_t_groups(dt2, 3)
    end
end