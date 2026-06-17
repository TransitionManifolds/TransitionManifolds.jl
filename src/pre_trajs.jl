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

# Sample `n` random points from `trajs`, excluding the end points.
# Returns a `(d, n)` Matrix.
function sample_points(trajs::Trajectories{T}, n::Int)::Matrix{T} where {T<:Real}
    # Build a flat index: (traj_index, point_index) for every point
    flat_index = [
        (i, j) for i in 1:(trajs.n_trajs) for j in 1:(size(trajs.trajs[i], 2) - 1)
    ]

    sampled = sample(flat_index, n; replace=false)

    out = Matrix{T}(undef, trajs.d, n)
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
    preprocess(data::Trajectories; anchors, dist=Euclidean(), max_dist::Real, min_samples::Int, max_samples::Int) -> PreprocessResult

Obtain approximate burst simulation data from [`Trajectories`](@ref).

The `anchors` should be provided as a `(d, n_anchors)` shaped `Matrix`.
Then, for each anchor ``a`` this function finds the trajectory points ``x_i`` closer than `max_dist`
in the metric `dist` (see `Distances.jl`), and adds the successors ``x_{i+1}`` to the samples for ``a``.
Only the `max_samples` closest trajectory points are considered for each anchor (default: `max_samples = ∞`).

If no `anchors` are provided, it uses random samples from the trajectories.
If no `max_dist` is provided, it uses half the average jump distance from the trajectories.

All `anchors` that by the end have less than `min_samples` samples are removed (default: `min_samples=1`).

The `res.info` dictionary contains

  - `res.info["anchors"]`: the final set of anchors
  - `res.info["max_dist"]`: the used `max_dist`
"""
function preprocess(
    data::Trajectories{T};
    anchors::Union{AbstractArray{T,2},Nothing}=nothing,
    dist::Metric=Euclidean(),
    max_dist::Union{Real,Nothing}=nothing,
    min_samples::Int=1,
    max_samples::Int=typemax(Int),
)::PreprocessResult where {T<:Real}
    # TODO: allow a vector of max_dists, one for each anchor.
    # We could also automatically guess a reasonable max_dist for each anchor
    # by taking the mean_jump_dist of the 10 nearest neighbors

    if isnothing(anchors)
        # if no anchors were provided, choose 1% random start points, but at most 1000
        n_anchors = round(Int, data.n_points * 0.01)
        n_anchors = clamp(n_anchors, 2, 1000)
        anchors = sample_points(data, n_anchors)
    end

    if isnothing(max_dist)
        max_dist = 0.5 * mean_jump_dist(data, dist)
    end

    size(anchors, 1) == data.d ||
        throw(ArgumentError("dimension `d` of trajs and anchors must match"))

    n_anchors = size(anchors, 2)

    # store views while collecting the samples to reduce allocations.
    # at the end the views are converted to owned data.
    sample_view = @view data.trajs[1][:, 1]
    V = typeof(sample_view)
    out = [V[] for _ in 1:n_anchors]

    for traj in data.trajs
        distances = pairwise(dist, @view(traj[:, 1:(end - 1)]), anchors; dims=2)

        for i in 1:n_anchors
            dcol = @view distances[:, i]
            valid_idxs = findall(<=(max_dist), dcol)
            n_valid = length(valid_idxs)
            if n_valid > max_samples
                partialsort!(valid_idxs, 1:max_samples; by=j -> dcol[j])
                n_valid = max_samples
            end
            for j in @view valid_idxs[1:n_valid]
                push!(out[i], @view traj[:, j + 1])
            end
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

    out = map(stack, out)  # this creates owned copies from the views

    return PreprocessResult(
        TransitionDistanceProblem(out), Dict("anchors" => anchors, "max_dist" => max_dist)
    )
end
