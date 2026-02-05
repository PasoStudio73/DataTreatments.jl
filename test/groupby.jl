using Test
using DataTreatments

using SoleData: Artifacts
using DataFrames, Random

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
natopsloader = Artifacts.NatopsLoader()
Xts, yts = Artifacts.load(natopsloader)

# ---------------------------------------------------------------------------- #
#                           windowing and treatment                            #
# ---------------------------------------------------------------------------- #
win = adaptivewindow(nwindows=3, overlap=0.2)
features = (mean, maximum)

rs_no_grp = DataTreatment(Xts, :reducesize; win)
ag_no_grp = DataTreatment(Xts, :aggregate; win, features)

# ---------------------------------------------------------------------------- #
#                                  groupby                                     #
# ---------------------------------------------------------------------------- #
rs_grp = DataTreatment(Xts, :reducesize; win, groups=(:vname,))
ag_grp = DataTreatment(Xts, :aggregate; win, features, groups=(:vname, :feat))

# macro groupby(x, idlabels...)
#     esc_idlabels = map(esc, idlabels)
#     quote
#         _x = $(esc(x))
#         _l = ($(esc_idlabels...),)
#         ds = _x.dataset
#         nid = length(_l)
        
#     #     # apply each window function to corresponding dimension
#     #     # if more dims than functions, reuse the last function
#     #     tuple((
#     #         let idx = min(i, length(_w))
#     #             _w[idx](dims[i])
#     #         end
#     #         for i in 1:length(dims)
#     #     )...)
#     end
# end

function groupby(d::DataTreatment, l::Symbol)

end

"""
    groupby(dt::DataTreatment, field::Symbol) -> Dict

Group feature indices by values in the specified `FeatureId` field.

# Arguments
- `dt::DataTreatment`: The data treatment object containing feature metadata
- `field::Symbol`: Field to group by (`:vname`, `:feat`, or `:nwin`)

# Returns
A dictionary mapping field values to vectors of feature indices.

# Examples
```julia
# Group by variable name
vname_groups = groupby(dt, :vname)
# Returns: Dict(:channel1 => [1,2,3,...], :channel2 => [10,11,12,...], ...)

# Group by feature function
feat_groups = groupby(dt, :feat)
# Returns: Dict(mean => [1,4,7,...], std => [2,5,8,...], ...)

# Group by window number
win_groups = groupby(dt, :nwin)
# Returns: Dict(1 => [1,2,3,...], 2 => [4,5,6,...], ...)
```
"""
function groupby(d::DataTreatment, field::Symbol)
    field in fieldnames(FeatureId) || 
        throw(ArgumentError("field must be one of $(fieldnames(FeatureId))"))

    getter = @eval $(Symbol(:get_, field))  
    featureids = get_featureid(d)
    feats = unique(getter.(featureids))

    groups = Vector{Vector{Int}}(undef, length(feats))

    for (i, f) in enumerate(feats)
        groups[i] = findall(fid -> getter(fid) == f, featureids)
    end

    return groups, feats
end

"""
    @groupby(dt, fields...)

Split a `DataTreatment` by multiple `FeatureId` fields, creating nested groups.

# Arguments
- `dt`: The `DataTreatment` object to group
- `fields...`: One or more `FeatureId` field symbols (`:vname`, `:feat`, `:nwin`)

# Returns
A nested structure where each level groups by the corresponding field in order.
- Single field: `(groups::Vector{Vector{Int}}, keys::Vector)`
- Multiple fields: Nested `Vector` of groups with corresponding keys at each level

# Examples
```julia
# Single field grouping
groups, keys = @groupby ag :vname

# Two-level grouping: first by vname, then by feat within each vname
groups, (vnames, feats) = @groupby ag :vname :feat

# Three-level grouping: vname -> feat -> nwin
groups, (vnames, feats, nwins) = @groupby ag :vname :feat :nwin

# Access specific group:
# groups[i] = features with vnames[i]
# groups[i][j] = features with vnames[i] and feats[j]
# groups[i][j][k] = indices with vnames[i], feats[j], nwins[k]
```
"""
macro groupby(dt, fields...)
    isempty(fields) && error("@groupby requires at least one field")
    
    dt_esc = esc(dt)
    
    length(fields) == 1 && begin
        field = esc(fields[1])
        return quote
            groupby($dt_esc, $field)
        end
    end

    # Build nested grouping
    quote
        let dt = $dt_esc
            first_groups, first_keys = groupby(dt, $(esc(fields[1])))
            
            nested_groups = Vector{Any}(undef, length(first_groups))
            all_nested_keys = nothing
            
            for (i, indices) in enumerate(first_groups)
                sub_featureids = dt.featureid[indices]
                
                # Group sub-features by remaining fields
                sub_groups, sub_keys = _groupby_nested(sub_featureids, $(esc.(fields[2:end])...))
                nested_groups[i] = sub_groups
                
                # Collect keys from first iteration
                if i == 1
                    all_nested_keys = sub_keys
                end
            end
            
            (nested_groups, (first_keys, all_nested_keys))
        end
    end
end

# Helper function for nested grouping with multiple fields
function _groupby_nested(featureids::Vector, field::Symbol, remaining_fields::Symbol...)
    getter = @eval $(Symbol(:get_, field))
    feats = unique(getter.(featureids))
    
    if isempty(remaining_fields)
        # Base case: single field
        groups = Vector{Vector{Int}}(undef, length(feats))
        for (i, f) in enumerate(feats)
            groups[i] = findall(fid -> getter(fid) == f, featureids)
        end
        return groups, feats
    else
        # Recursive case: multiple fields
        nested_groups = Vector{Any}(undef, length(feats))
        all_nested_keys = nothing
        
        for (i, f) in enumerate(feats)
            sub_indices = findall(fid -> getter(fid) == f, featureids)
            sub_featureids = featureids[sub_indices]
            
            sub_groups, sub_keys = _groupby_nested(sub_featureids, remaining_fields...)
            nested_groups[i] = sub_groups
            
            if i == 1
                all_nested_keys = sub_keys
            end
        end
        
        return nested_groups, (feats, all_nested_keys)
    end
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

fields = [:vname, :feat]

# initial setup Vector{Vector} of all indexes and featureids
featureids = [get_featureid(d)]
idxs = [[1:length(first(f))...]]

_groupby(idxs[1], featureids[1], fields[1])

function test(idxs::Vector{Vector{Int64}}, featureids::Vector{Vector{FeatureId}}, fields::Vector{Symbol})
    length(fields) == 1 && return _groupby(first(idxs), first(featureids), first(fields))

    ngroups = length(featureids)
    groups = []
    feats = []

    a, b = test(idxs, featureids, fields[1:end-1])
    push!(groups, a)
    push!(feats, b)

    return groups, feats
end

function atest(idxs::Vector{Vector{Int64}}, featureids::Vector{Vector{FeatureId}}, fields::Vector{Symbol})
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
        all_groups[i], all_feats[i] = atest(sub_idxs, sub_featureids, fields[2:end])
    end

    return all_groups, all_feats
end

macro atest(idxs, featureids, fields...)
    isempty(fields) && error("@atest requires at least one field")
    quote
        atest($(esc(idxs)), $(esc(featureids)), [$(esc.(fields)...)] )
    end
end