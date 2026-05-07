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
    TransitionDistanceResult{T<:Real}

Struct for storing the result of the transition distances computation,
see [`compute_distances`](@ref).
"""
struct TransitionDistanceResult{T<:Real}
    distances::Matrix{T}
    info::Dict{String,<:Any}
end

"""
    compute_distances(data::AbstractArray{<:Real,3}, alg::AbstractTransitionDistanceAlgorithm; progress=false) -> TransitionDistanceResult

Compute pairwise distances of transition density functions from `data`, returning a [`TransitionDistanceResult`](@ref) object `res`.

The `data` should contain the endpoints of `n_samples` burst simulations for `n_anchor` anchor points,
and have the shape `(d, n_samples, n_anchors)`.
The distance between transition densities ``p_x`` and ``p_y`` for each pair of anchor points ``x`` and ``y``
is estimated from the `data` using the specified algorithm `alg`.

The result `res` contains the pairwise distance matrix `res.distances`
and the `res.info` dictionary, which is used to store further information,
see the documentation for each specific algorithm.
"""
function compute_distances(
    data::AbstractArray{<:Real,3},
    alg::AbstractTransitionDistanceAlgorithm;
    progress::Bool=false,
)::TransitionDistanceResult
    error("No implementation of `compute_distances` for algorithm `$(typeof(alg))`")
end

"""
    EmbeddingResult{T<:Real}

Struct for storing the result of the embedding computation,
see [`compute_embedding`](@ref).
"""
struct EmbeddingResult{T<:Real}
    coordinates::Matrix{T}
    info::Dict{String,<:Any}
end

"""
    compute_embedding(distances::AbstractMatrix{<:Real}, alg::AbstractEmbeddingAlgorithm; n_coordinates, progress=false) -> EmbeddingResult

Compute an embedding from `distances`, returning a [`EmbeddingResult`](@ref) object `res`.

The `distances` should be a `(n_anchors, n_anchors)` symmetric distance matrix with 0 on the diagonal.

The desired dimension of the embedding can be specified using `n_coordinates`,
but it is not guaranteed that the `alg` is able to provide exactly this number.
By default, all computed coordinates are returned.

The result `res` contains the `res.coordinates` of the embedding in the shape `(n_anchors, n_coordinates)`,
and the `res.info` dictionary, which is used to store further information,
see the documentation for each specific algorithm.
"""
function compute_embedding(
    distances::AbstractMatrix{<:Real},
    alg::AbstractEmbeddingAlgorithm;
    n_coordinates::Int=typemax(Int),
    progress::Bool=false,
)::EmbeddingResult
    error("No implementation of `compute_embedding` for algorithm `$(typeof(alg))`")
end

"""
    compute_embedding(dres::TransitionDistanceResult, alg::AbstractEmbeddingAlgorithm; kwargs...) -> EmbeddingResult

A [`TransitionDistanceResult`](@ref) `dres` can be provided instead of a distance matrix `distances`.
"""
compute_embedding(
    dres::TransitionDistanceResult, alg::AbstractEmbeddingAlgorithm; kwargs...
) = compute_embedding(dres.distances, alg; kwargs...)

"""
    compute_transition_manifold(data::AbstractArray{<:Real,3}, distance_alg::AbstractTransitionDistanceAlgorithm, embedding_alg::AbstractEmbeddingAlgorithm; n_coordinates, progress::Bool=false) -> Tuple{TransitionDistanceResult,EmbeddingResult}

Compute an embedding using the `embedding_alg` from the transition distances that are calculated using the `distance_alg`.

This is an auxiliary function that simply first calls [`compute_distances`](@ref) and then [`compute_embedding`](@ref).

The `data` should contain the endpoints of `n_samples` burst simulations for `n_anchor` anchor points,
and have the shape `(d, n_samples, n_anchors)`.
"""
function compute_transition_manifold(
    data::AbstractArray{<:Real,3},
    distance_alg::AbstractTransitionDistanceAlgorithm,
    embedding_alg::AbstractEmbeddingAlgorithm;
    n_coordinates::Int=typemax(Int),
    progress::Bool=false,
)::Tuple{TransitionDistanceResult,EmbeddingResult}
    dres = compute_distances(data, distance_alg; progress=progress)
    eres = compute_embedding(
        dres, embedding_alg; n_coordinates=n_coordinates, progress=progress
    )
    return (dres, eres)
end
