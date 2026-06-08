# Preprocessing of trajectory data

"""
    Trajectories{T<:Real}(trajs)

Struct for storing one or multiple trajectories `trajs` of data type `T`.

The argument `trajs` is either a single trajectory in the form of a `(d, n_points)` shaped `Matrix{T}`,
or multiple trajectories in the form of a `Vector{Matrix{T}}`.
"""
struct Trajectories{T<:Real}
    trajs::AbstractVector{<:AbstractArray{T,2}}
    n_trajs::Int  # number of trajectories
    d::Int  # dimension
    n_points::Int  # total number of points

    function Trajectories(trajs::AbstractVector{<:AbstractArray{T,2}}) where {T<:Real}
        n_trajs = length(trajs)
        n_trajs > 0 || throw(ArgumentError("trajs must not be empty"))
        d = size(trajs[1], 1)
        n_points = 0
        for traj in trajs
            size(traj, 1) == d ||
                throw(ArgumentError("all trajs must have same dimension `d`"))
            size(traj, 2) >= 2 ||
                throw(ArgumentError("all trajs must contain atleast two points"))
            n_points += size(traj, 2)
        end
        new{T}(trajs, n_trajs, d, n_points)
    end
end

Trajectories(traj::AbstractArray{<:Real,2}) = Trajectories([traj])

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
    preprocess(data::Trajectories; anchors, dist=Euclidean(), max_dist::Real) -> PreprocessResult

Obtain approximate burst simulation data from [`Trajectories`](@ref).

The `anchors` should be provided as a `(d, n_anchors)` shaped `Matrix`.
Then, for each anchor ``a`` this function finds the trajectory points ``x_i`` closer than `max_dist`
in the metric `dist` (see `Distances.jl`), and adds the successors ``x_{i+1}`` to the samples for ``a``.

If no `anchors` are provided, it uses random samples from the trajectories.
If no `max_dist` is provided, it uses half the average jump distance from the trajectories.

Any `anchors` that by the end have no samples are removed, so that the result does not contain empty arrays.

The `res.info` dictionary contains

  - `res.info["anchors"]`: the final set of anchors
  - `res.info["max_dist"]`: the used `max_dist`
"""
function preprocess(
    data::Trajectories{T};
    anchors::Union{AbstractArray{T,2},Nothing}=nothing,
    dist::Metric=Euclidean(),
    max_dist::Union{Real,Nothing}=nothing,
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
        @views distances = pairwise(dist, traj[:, 1:(end - 1)], anchors; dims=2)

        for i in 1:n_anchors
            for j in axes(distances, 1)
                if distances[j, i] <= max_dist
                    @views push!(out[i], traj[:, j + 1])
                end
            end
        end
    end

    # deal with anchors that dont have any matching samples
    not_empty_idxs = findall(s -> !isempty(s), out)
    n_empty = n_anchors - length(not_empty_idxs)
    if n_empty == n_anchors
        error("did not find any matching samples for any provided anchor")
    end
    n_empty == 0 ||
        @warn "$n_empty anchors do not have any matching samples and were removed. See the `res.info` dict for the remaining anchors"
    filter!(s -> !isempty(s), out)

    out = map(stack, out)  # this creates owned copies from the views
    anchors = anchors[:, not_empty_idxs] # remove the anchors that have no samples

    return PreprocessResult(
        TransitionDistanceProblem(out), Dict("anchors" => anchors, "max_dist" => max_dist)
    )
end
