module DataTreatments

using Statistics

export movingwindow, wholewindow, splitwindow, adaptivewindow
export @evalwindow
include("slidingwindow.jl")

export applyfeat, aggregate, reducesize
include("treatment.jl")

end
