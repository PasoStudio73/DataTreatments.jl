# ---------------------------------------------------------------------------- #
#                                  groupby                                     #
# ---------------------------------------------------------------------------- #
macro groupby(d, fields...)
    isempty(fields) && error("@atest requires at least one field")

    quote
        df = $(esc(d))
        # initial setup Vector{Vector} of all indexes and featureids
        featureids = get_featureid(df)
        idxs = [[1:length(featureids)...]]
        @show idxs
        @show featureids
        _groupby(idxs, [featureids], [$(esc.(fields)...)])
    end
end

function _groupby(idxs::Vector{Vector{Int64}}, featureids::Vector{Vector{FeatureId}}, fields::Vector{Symbol})
    if length(fields) == 1
        ngroups = length(featureids)
        all_groups = Vector{Vector{Vector{Int}}}(undef, ngroups)
        all_feats = Vector{Vector{Vector{FeatureId}}}(undef, ngroups)

        for i in 1:ngroups
            all_groups[i], all_feats[i] = _groupby(idxs[i], featureids[i], fields[1])
        end

        # flatten one level
        return vcat(all_groups...), vcat(all_feats...)
    end

    ngroups = length(featureids)
    all_groups = Vector{Any}(undef, ngroups)
    all_feats = Vector{Any}(undef, ngroups)

    for i in 1:ngroups
        sub_idxs, sub_featureids = _groupby(idxs[i], featureids[i], fields[1])
        all_groups[i], all_feats[i] = _groupby(sub_idxs, sub_featureids, fields[2:end])
    end

    return all_groups, all_feats
end

function _groupby(idxs::Vector{Int64}, featureids::Vector{FeatureId}, field::Symbol)
    getter = @eval $(Symbol(:get_, field))  
    feats = unique(getter.(featureids))

    groups = Vector{Vector{Int}}(undef, length(feats))
    feat_groups = Vector{Vector{FeatureId}}(undef, length(feats))

    for (i, f) in enumerate(feats)
        groups[i] = idxs[findall(fid -> getter(fid) == f, featureids)]
        feat_groups[i] = featureids[findall(fid -> getter(fid) == f, featureids)]
    end

    return groups, feat_groups
end