# ---------------------------------------------------------------------------- #
#                            TargetStructure struct                            #
# ---------------------------------------------------------------------------- #
"""
    TargetStructure

A structure used by `DataTreatments` to store information about the target (dependent variable) of a dataset.
It holds both the vector of target values and, for classification tasks, 
the labels associated with discrete-encoded classes.

This struct is constructed automatically from a target vector and is used internally by `DataTreatment` objects.

# Fields
- `values::Vector{Union{<:Int, <:AbstractFloat}}`: 
  The encoded target values (integers for classification via `discrete_encode`, floats for regression).
- `labels::Union{Nothing, CategoricalArrays.CategoricalVector}`: 
  The class labels for classification tasks (returned by `discrete_encode`), or `nothing` for regression.

# Constructor

    TargetStructure(y::AbstractVector) -> TargetStructure

If `eltype(y) <: AbstractFloat`, stores `y` directly as values with `labels = nothing` (regression).
Otherwise, calls `discrete_encode(y)` to produce integer-encoded values 
and the corresponding categorical labels (classification).
"""
struct TargetStructure{T}
    values::Vector{T}
    labels::Union{Nothing,CategoricalArrays.CategoricalVector}

    function TargetStructure(y::AbstractVector)
        T = eltype(y)
        if T <: AbstractFloat 
            new{T}(y, nothing)
        else
            y, l = discrete_encode(y)
            new{eltype(y)}(y, l)
        end
    end
end

Base.length(ts::TargetStructure) = length(ts.values)

# ---------------------------------------------------------------------------- #
#                               getter methods                                 #
# ---------------------------------------------------------------------------- #
"""
get_values(ts::TargetStructure)

Returns the vector of encoded target values.
"""
get_values(ts::TargetStructure) = ts.values

"""
get_labels(ts::TargetStructure)

Returns the class labels for classification tasks, or nothing for regression.
"""
get_labels(ts::TargetStructure) = ts.labels

