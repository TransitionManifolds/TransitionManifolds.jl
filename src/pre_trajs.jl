# Preprocessing of trajectory data

"""
    Trajectories{T<:Real}(data)

Struct for storing one or multiple trajectories `trajs` of data type `T`.

The `data` is either a single trajectory in the form of a `(d, n_points)` shaped `Matrix{T}`,
or multiple trajectories in the form of a `Vector{Matrix{T}}` where each trajectory has the same dimension `d`.

The `length(trajs)` returns the total number of points across all trajectories.
Also supports iteration over all trajectory points, e.g., `for point in trajs`.

Indexing at `i` returns a view of the `i`-th point, `1 <= i <= length(trajs)`.
(Example: if the first trajectory has 3 points, `trajs[4]` returns the first point of the second trajectory.)
"""
struct Trajectories{T<:Real}
    trajs::AbstractVector{<:AbstractArray{T,2}}
    n_trajs::Int  # number of trajectories
    d::Int  # dimension
    n_points::Int  # total number of points

    # start inidices of each trajectory for indexing,
    # e.g., trajectory `i` has indices `offsets[i]:offsets[i+1]-1`
    offsets::Vector{Int}

    function Trajectories(trajs::AbstractVector{<:AbstractArray{T,2}}) where {T<:Real}
        n_trajs = length(trajs)
        n_trajs > 0 || throw(ArgumentError("trajs must not be empty"))
        d = size(trajs[1], 1)
        n_points = 0
        offsets = Int[1]
        for traj in trajs
            size(traj, 1) == d ||
                throw(ArgumentError("all trajs must have same dimension `d`"))
            this_n_points = size(traj, 2)
            this_n_points >= 2 ||
                throw(ArgumentError("all trajs must contain atleast two points"))
            n_points += this_n_points
            push!(offsets, offsets[end] + this_n_points)
        end
        new{T}(trajs, n_trajs, d, n_points, offsets)
    end
end

Trajectories(traj::AbstractArray{<:Real,2}) = Trajectories([traj])

Base.length(trajs::Trajectories) = trajs.n_points

function Base.iterate(trajs::Trajectories, state=(1, 1))
    traj_idx, point_idx = state
    traj_idx <= trajs.n_trajs || return nothing

    traj = trajs.trajs[traj_idx]
    out = @view traj[:, point_idx]
    new_state = (point_idx < size(traj, 2)) ? (traj_idx, point_idx + 1) : (traj_idx + 1, 1)
    return (out, new_state)
end

function Base.getindex(trajs::Trajectories, i::Int)
    1 <= i <= trajs.n_points || throw(BoundsError(trajs, i))
    traj_idx = searchsortedlast(trajs.offsets, i)
    point_idx = i - trajs.offsets[traj_idx] + 1
    return @view trajs.trajs[traj_idx][:, point_idx]
end
Base.getindex(trajs::Trajectories, idxs::AbstractVector{Int}) = [trajs[i] for i in idxs]
Base.firstindex(::Trajectories) = 1
Base.lastindex(trajs::Trajectories) = trajs.n_points

"""
    is_endpoint(trajs::Trajectories, i::Int) -> Bool

Whether `trajs[i]` is the endpoint of a trajectory.

If `is_endpoint(trajs, i)` is true, then `trajs[i+1]` is the startpoint of the next trajectory.
"""
function is_endpoint(trajs::Trajectories, i::Int)::Bool
    1 <= i <= trajs.n_points || throw(BoundsError(trajs, i))
    return insorted(i + 1, trajs.offsets)
end

"""
    sample_points(trajs::Trajectories{T}, k::Int) -> Matrix{T}

Sample `k` random points from `trajs`, excluding the end points.

Returns a `(d, k)` matrix, where `d` is the dimension of `trajs`.
"""
function sample_points(trajs::Trajectories{T}, k::Int)::Matrix{T} where {T<:Real}
    # Build a flat index: (traj_index, point_index) for every point
    flat_index = [
        (i, j) for i in 1:(trajs.n_trajs) for j in 1:(size(trajs.trajs[i], 2) - 1)
    ]

    sampled = sample(flat_index, k; replace=false)

    out = Matrix{T}(undef, trajs.d, k)
    for (k, (i, j)) in enumerate(sampled)
        out[:, k] = trajs.trajs[i][:, j]
    end
    return out
end

# Compute the average jump distance from a trajectory point to its successor.
function mean_jump_dist(trajs::Trajectories, dist::Metric)::Float64
    mjd = 0.0
    for traj in trajs.trajs
        for i in 1:(size(traj, 2) - 1)
            @views mjd += dist(traj[:, i], traj[:, i + 1])
        end
    end
    return mjd / (trajs.n_points - trajs.n_trajs)
end

"""
    FarthestPointSamplingResult

Struct for storing the result of a farthest point sampling computation,
see [`farthest_point_sampling`](@ref).
"""
struct FarthestPointSamplingResult
    selected::Vector{Int}  # indices of selected points
    assignments::Vector{Int}  # `assignments[i] = l` <=> selected point closest to `i` is `selected[l]`
end

"""
    farthest_point_sampling(trajs::Trajectories, k::Int; dist::Metric=Euclidean(), init_idx=1, centering=false) -> FarthestPointSamplingResult

Select `k` points from `trajs` using farthest point sampling.

Starting from the point `trajs[init_idx]`, iteratively add the point from `trajs` to the selected points
that is farthest from any other selected point with respect to `dist`.

Optionally, if `centering=true`, execute a postprocessing step which returns the most central point of each cluster
instead of the farthest points picked initially.
The cluster `l` is defined by all points for which the closest selected point is the `l`-th selected point.
The central point of cluster `l` has the minimal sum of distances to all other points in the cluster.

Returns a [`FarthestPointSamplingResult`](@ref) object `res` that contains

  - `res.selected`: the indices of the `k` selected points. Retrieve the points via `trajs[res.selected]`.
  - `res.assignments`: the cluster assignment of each point, i.e., `assignments[i] = l` <=> selected point closest to `i` is `selected[l]`.
"""
function farthest_point_sampling(
    trajs::Trajectories,
    k::Int;
    dist::Metric=Euclidean(),
    init_idx::Int=1,
    centering::Bool=false,
)::FarthestPointSamplingResult
    res = _farthest_point_sampling(trajs, k, dist, init_idx)
    if centering
        res = _center_farthest_points(res, trajs, dist)
    end
    return res
end

function _farthest_point_sampling(
    trajs::Trajectories, k::Int, dist::Metric, init_idx::Int
)::FarthestPointSamplingResult
    n = length(trajs)
    1 <= init_idx <= n ||
        throw(ArgumentError("the `init_idx` must be between 1 and `n_points`=$n"))

    # indices of selected points
    selected = Vector{Int}(undef, k)
    selected[1] = init_idx

    # at the start all points are assigned to cluster 1
    assignments = fill(1, n)

    # distance of each point `i` to the nearest selected point `assignments[i]`
    dmin = Vector{Float64}(undef, n)
    this_point = trajs[init_idx]
    for (i, point) in enumerate(trajs)
        dmin[i] = dist(this_point, point)
    end

    # farthest point sampling
    @views for l in 2:k
        next_idx = argmax(dmin)
        selected[l] = next_idx

        # update distances
        this_point = trajs[next_idx]
        for (i, point) in enumerate(trajs)
            new_dist = dist(this_point, point)
            if new_dist < dmin[i]
                dmin[i] = new_dist
                assignments[i] = l
            end
        end
    end

    return FarthestPointSamplingResult(selected, assignments)
end

# replace each selected point by the point in its cluster that minimizes
# the cumulative distance to all other cluster points
function _center_farthest_points(
    res::FarthestPointSamplingResult, trajs::Trajectories, dist::Metric
)::FarthestPointSamplingResult
    k = length(res.selected)
    n = length(res.assignments)

    # `clusters[l]` are the points assigned to cluster `l`
    clusters = [Int[] for _ in 1:k]
    for i in 1:n
        push!(clusters[res.assignments[i]], i)
    end

    # indices of selected points
    selected = Vector{Int}(undef, k)

    # select most central point in each cluster
    for l in 1:k
        cluster_idxs = clusters[l]
        cluster_data = trajs[cluster_idxs]

        cum_dists = zeros(length(cluster_data))
        for i in eachindex(cluster_data)
            this_point = cluster_data[i]
            for j in eachindex(cluster_data)
                if i == j
                    continue
                end
                cum_dists[i] += dist(this_point, cluster_data[j])
            end
        end
        selected[l] = cluster_idxs[argmin(cum_dists)]
    end

    return FarthestPointSamplingResult(selected, res.assignments)
end

"""
    preprocess(data::Trajectories; anchors, dist=Euclidean(), max_dist, min_samples::Int, max_samples::Int) -> PreprocessResult

Obtain approximate burst simulation data from [`Trajectories`](@ref).

The `anchors` should be provided either as a `(d, n_anchors)` shaped `Matrix`, or as an `Int`.
If given as an `Int`, this number of anchors will be generated using [`farthest_point_sampling`](@ref).

Then, for each anchor ``a`` this function finds the trajectory points ``x_i`` closer than `max_dist`
in the metric `dist` (see `Distances.jl`), and adds the successors ``x_{i+1}`` to the samples for ``a``.
Only the `max_samples` closest trajectory points are considered for each anchor (default: `max_samples = ∞`).

The `max_dist` can either be given as a `Real`, in which case this value will be used for all anchors,
or as a `Vector{Real}`, in which case `max_dist[l]` will be used for the `l`-th anchor.
If no `max_dist` is provided, it uses half the average jump distance from the trajectories.

All `anchors` that by the end have less than `min_samples` samples are removed (default: `min_samples=1`).

The `res.info` dictionary contains

  - `res.info["anchors"]`: the final set of anchors
  - `res.info["max_dist"]`: the used `max_dist` for each anchor
"""
function preprocess(
    data::Trajectories{T};
    anchors::Union{AbstractArray{T,2},Int,Nothing}=nothing,
    dist::Metric=Euclidean(),
    max_dist::Union{Real,Vector{<:Real},Nothing}=nothing,
    min_samples::Int=1,
    max_samples::Int=typemax(Int),
)::PreprocessResult where {T<:Real}
    # TODO: automatically guess a reasonable max_dist for each anchor
    # by taking the mean_jump_dist of the k nearest neighbors

    # TODO: allow PreMetric or SemiMetric?

    # process `anchors`
    if isnothing(anchors)
        # if no anchors were provided, set it to 1% of trajs points, but at most 1000
        anchors = round(Int, length(data) * 0.01)
        anchors = clamp(anchors, 2, 1000)
    end
    if anchors isa Int
        # use farthest point sampling to generate anchors
        anchors >= 2 || throw(ArgumentError("`anchors` must be at least 2"))
        res = farthest_point_sampling(data, anchors; dist=dist, centering=true)
        anchors = stack(data[res.selected])
    end
    size(anchors, 1) == data.d ||
        throw(ArgumentError("dimension `d` of trajs and anchors must match"))
    n_anchors = size(anchors, 2)
    # at this point `anchors` is a (d, n_anchors) matrix

    # process `max_dist`
    if isnothing(max_dist)
        max_dist = 0.5 * mean_jump_dist(data, dist)
    end
    if max_dist isa Real
        max_dist = fill(max_dist, n_anchors)
    end
    length(max_dist) == n_anchors || throw(
        ArgumentError("`max_dist` must have the same length as the number of anchors")
    )
    # at this point `max_dist` is a (n_anchors,) vector

    # store views of points while collecting the samples to reduce allocations.
    # at the end the views are converted to owned data.
    sample_view = data[1]
    V = typeof(sample_view)
    out = [V[] for _ in 1:n_anchors]

    # compute distances between anchors and trajs
    # TODO: Switch dims of distances?
    # Currently, it is the wrong way around during construction in anchor_trajs_distances,
    # but the correct way around for accessing dcol later.
    distances = anchor_trajs_distances(anchors, data, dist)
    # the end point of a trajectory is not valid
    for offset in @view(data.offsets[2:end])
        # offset - 1 is the last index of a trajectory
        distances[offset - 1, :] .= typemax(Float64)
    end

    # find the matching samples
    for i in 1:n_anchors
        dcol = @view distances[:, i]
        # because the end point of each trajectory has dcol=Inf,
        # an end point is never valid
        valid_idxs = findall(<=(max_dist[i]), dcol)
        n_valid = length(valid_idxs)
        if n_valid > max_samples
            partialsort!(valid_idxs, 1:max_samples; by=j -> dcol[j])
            n_valid = max_samples
        end

        sizehint!(out[i], n_valid)
        for j in @view valid_idxs[1:n_valid]
            # for each valid j, push the successor j+1
            push!(out[i], data[j + 1])
        end
    end

    # remove anchors that have less than `min_samples` samples
    keep_idxs = findall(s -> length(s) >= min_samples, out)
    n_remove = n_anchors - length(keep_idxs)
    if n_remove == n_anchors
        error(
            "all anchors have less than `min_samples` matching samples and have been removed",
        )
    end
    n_remove == 0 ||
        @warn "$n_remove anchors have less than `min_samples` matching samples and were removed. See the `res.info` dict for the remaining anchors"
    filter!(s -> length(s) >= min_samples, out)
    anchors = anchors[:, keep_idxs]
    max_dist = max_dist[keep_idxs]

    out = map(stack, out)  # this creates owned copies from the views

    return PreprocessResult(
        TransitionDistanceProblem(out), Dict("anchors" => anchors, "max_dist" => max_dist)
    )
end

# For `anchors` of shape (d, n_anchors) and `trajs` containing n_points points,
# compute the (n_points, n_anchors) pairwise distance matrix.
function anchor_trajs_distances(
    anchors::AbstractMatrix{T}, trajs::Trajectories{T}, dist::Metric
)::Matrix{Float64} where {T<:Real}
    n_anchors = size(anchors, 2)
    distances = Matrix{Float64}(undef, length(trajs), n_anchors)

    for (i, traj) in enumerate(trajs.trajs)
        start_idx = trajs.offsets[i]
        end_idx = trajs.offsets[i + 1] - 1
        @views pairwise!(dist, distances[start_idx:end_idx, :], traj, anchors; dims=2)
    end

    return distances
end
