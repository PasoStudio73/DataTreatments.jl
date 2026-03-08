# ---------------------------------------------------------------------------- #
#                              errors handling                                 #
# ---------------------------------------------------------------------------- #
function validate_vector_lengths(vectors::AbstractVector...)
    """
    Validate that all vectors have the same length.
    
    Args:
        vectors: AbstractVector...
        
    Throws:
        DimensionMismatch: If vectors have different lengths
    """
    isempty(vectors) && return
    
    reference_length = length(first(vectors))
    
    for (i, vec) in enumerate(vectors)
        length(vec) != reference_length &&
            throw(DimensionMismatch(
                "Vector $i has length $(length(vec)), expected $reference_length"
            ))
    end
end