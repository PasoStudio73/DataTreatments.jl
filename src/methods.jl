using DataTreatments
const DT = DataTreatments
TreatmentOutput = DT.TreatmentOutput

using DataFrames
using Random
using CategoricalArrays

function create_image(seed::Int; n=6)
    Random.seed!(seed)
    rand(Float64, n, n)
end

function build_test_df()
    DataFrame(
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
        ts3 = [[1.0, 1.2, 1.2, 2.6, NaN, 4.0, 4.2], NaN, NaN, missing, [3.0, NaN, 4.4, missing, 5.8, 7.0, 7.2]],
        ts4 = [[6.0, 5.2, missing, 4.4, 1.2, 3.6, 2.8], missing, [5.0, 4.2, NaN, 3.4, missing, 2.6, 1.8], [8.0, 7.2, missing, 6.4, NaN, 5.6, 4.8], [9.0, NaN, 8.2, missing, 7.4, 6.6, 5.8]],
        img1 = [create_image(i) for i in 1:5],
        img2 = [i == 1 ? NaN : create_image(i+10) for i in 1:5],
        img3 = [create_image(i+20) for i in 1:5],
        img4 = [i == 3 ? missing : create_image(i+30) for i in 1:5]
    )
end

# ---------------------------------------------------------------------------- #
#                               getter methods                                 #
# ---------------------------------------------------------------------------- #
get_discrete(dt::TreatmentOutput) = filter(d -> d isa DiscreteDataset, dt)
get_continuous(dt::TreatmentOutput) = filter(d -> d isa ContinuousDataset, dt)
get_multidim(dt::TreatmentOutput) = filter(d -> d isa MultidimDataset, dt)

get_aggregated(dt::TreatmentOutput) = filter(d -> d isa MultidimDataset && 
    # isa(getfield(d, :info), Vector) && 
    # !isempty(getfield(d, :info)) && 
    eltype(getfield(d, :info)) <: AggregateFeat, dt)

get_reduced(dt::TreatmentOutput) = filter(d -> d isa MultidimDataset && 
    # isa(getfield(d, :info), Vector) && 
    # !isempty(getfield(d, :info)) && 
    eltype(getfield(d, :info)) <: ReduceFeat, dt)

# ---------------------------------------------------------------------------- #
#                             get tabular method                               #
# ---------------------------------------------------------------------------- #
# args:
# treatments::Vararg{Base.Callable}=TreatmentGroup(aggrfunc=DefaultAggrFunc,)
# kwargs:
# treatment_ds::Bool=true,
# leftover_ds::Bool=true,

function get_tabular(
    dt::DataTreatment,
    args...;
    groupby_split::Bool=false,
    output_type::Base.Callable=standard, # standard, matrix, dataframe
    kwargs...
) # ::Vector{Union{AbstractDataset,AbstractMatrix,DataFrame}}
    data = get_dataset(dt::DataTreatment, args...; kwargs...)

    get_discrete(data)
    get_continuous(data)
    get_aggregated(data)

    return output_type(data, groupby_split)
end

# ---------------------------------------------------------------------------- #
#                            get multidim method                               #
# ---------------------------------------------------------------------------- #
function get_multidim(
    dt::DataTreatment,
    treatments::Vararg{Base.Callable}=TreatmentGroup(aggrfunc=DefaultAggrFunc,);
    # treatment_ds::Bool=true,
    leftover_ds::Bool=true,
    # groupby_split::Bool=false,
    output_type::Symbol=:standard # :standard, :matrix, :dataframe
) # ::Vector{Union{AbstractDataset,AbstractMatrix,DataFrame}}

end

df = build_test_df()
dt = DataTreatment(df)

# result = get_tabular(
data = get_tabular(
    dt,
    TreatmentGroup(
        name_expr=["V3", "V4", "V5"]
    ),
    TreatmentGroup(
        dims=1,
        aggrfunc=DT.aggregate(
            features=(mean, maximum),
            win=(DT.adaptivewindow(nwindows=5, overlap=0.4),)
        )),
    TreatmentGroup(
        dims=2,
        aggrfunc=DT.reducesize()
    ),
    # output_type=dataframe
)

@show result isa DT.TreatmentOutput