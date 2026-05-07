# ------------- Abstract Interface -----------------
"""
    AbstractTransitionDistanceAlgorithm

Abstract type for algorithms for computing the distance between transition density functions.

See also [`compute_distances`](@ref).
"""
abstract type AbstractTransitionDistanceAlgorithm end

"""
    AbstractEmbeddingAlgorithm

Abstract type for algorithms for computing low-dimensional embeddings.

See also [`compute_embedding`](@ref).
"""
abstract type AbstractEmbeddingAlgorithm end

"""
    TransitionDistanceSolution{T<:Real}

Struct for storing the solution of the transition distances computation,
see [`compute_distances`](@ref).
"""
struct TransitionDistanceSolution{T<:Real}
    distances::Matrix{T}
    info::Dict{String,<:Any}
end

"""
    compute_distances(data::AbstractArray{<:Real,3}, alg::AbstractTransitionDistanceAlgorithm; progress=false) -> TransitionDistanceSolution

Compute pairwise distances of transition density functions from `data`, returning a [`TransitionDistanceSolution`](@ref) object `sol`.

The `data` should contain the endpoints of `n_samples` burst simulations for `n_anchor` anchor points,
and have the shape `(d, n_samples, n_anchors)`.
The distance between transition densities ``p_x`` and ``p_y`` for each pair of anchor points ``x`` and ``y``
is estimated from the `data` using the specified algorithm `alg`.

The solution `sol` contains the pairwise distance matrix `sol.distances`
and the `sol.info` dictionary, which is used to store further information,
see the documentation for each specific algorithm.
"""
function compute_distances(
    data::AbstractArray{<:Real,3},
    alg::AbstractTransitionDistanceAlgorithm;
    progress::Bool=false,
)::TransitionDistanceSolution
    error("No implementation of `compute_distances` for algorithm `$(typeof(alg))`")
end

"""
    EmbeddingSolution{T<:Real}

Struct for storing the solution of the embedding computation,
see [`compute_embedding`](@ref).
"""
struct EmbeddingSolution{T<:Real}
    coordinates::Matrix{T}
    info::Dict{String,<:Any}
end

"""
    compute_embedding(distances::AbstractMatrix{<:Real}, alg::AbstractEmbeddingAlgorithm; n_coordinates, progress=false) -> EmbeddingSolution

Compute an embedding from `distances`, returning a [`EmbeddingSolution`](@ref) object `sol`.

The `distances` should be a `(n_anchors, n_anchors)` symmetric distance matrix with 0 on the diagonal.

The desired dimension of the embedding can be specified using `n_coordinates`,
but it is not guaranteed that the `alg` is able to provide exactly this number.
By default, all computed coordinates are returned.

The solution `sol` contains the `sol.coordinates` of the embedding in the shape `(n_anchors, n_coordinates)`,
and the `sol.info` dictionary, which is used to store further information,
see the documentation for each specific algorithm.
"""
function compute_embedding(
    distances::AbstractMatrix{<:Real},
    alg::AbstractEmbeddingAlgorithm;
    n_coordinates::Int=typemax(Int),
    progress::Bool=false,
)::EmbeddingSolution
    error("No implementation of `compute_embedding` for algorithm `$(typeof(alg))`")
end

"""
    compute_embedding(dsol::TransitionDistanceSolution, alg::AbstractEmbeddingAlgorithm; kwargs...) -> EmbeddingSolution

A [`TransitionDistanceSolution`](@ref) `dsol` can be provided instead of a distance matrix `distances`.
"""
compute_embedding(
    dsol::TransitionDistanceSolution, alg::AbstractEmbeddingAlgorithm; kwargs...
) = compute_embedding(dsol.distances, alg; kwargs...)

function compute_transition_manifold(
    data::AbstractArray{<:Real,3},
    distance_alg::AbstractTransitionDistanceAlgorithm,
    embedding_alg::AbstractEmbeddingAlgorithm;
    n_coordinates::Int=typemax(Int),
    progress::Bool=false,
)::Tuple{TransitionDistanceSolution,EmbeddingSolution}
    dsol = compute_distances(data, distance_alg; progress=progress)
    esol = compute_embedding(
        dsol, embedding_alg; n_coordinates=n_coordinates, progress=progress
    )
    return (dsol, esol)
end
