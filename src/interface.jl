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

const ContiguousData{T<:Real} = AbstractArray{T,3}
const JaggedData{T<:Real} = AbstractVector{<:AbstractArray{T,2}}

const ContiguousWeights{W<:Real} = AbstractArray{W,2}
const JaggedWeights{W<:Real} = AbstractVector{<:AbstractVector{W}}

"""
    AbstractDataLayout

Abstract type for the layout of data.

See also [`TransitionDistanceProblem`](@ref) and [`compute_distances`](@ref).
"""
abstract type AbstractDataLayout end

"""
    Contiguous <: AbstractDataLayout

Contiguous data layout `Array{T, 3}`.

See also [`TransitionDistanceProblem`](@ref) and [`compute_distances`](@ref).
"""
struct Contiguous <: AbstractDataLayout end

"""
    Jagged <: AbstractDataLayout

Jagged data layout `Vector{Array{T,2}}`.

See also [`TransitionDistanceProblem`](@ref) and [`compute_distances`](@ref).
"""
struct Jagged <: AbstractDataLayout end

"""
    TransitionDistanceProblem{T,W,L}(data; weights)

Struct for holding the data that is used to compute transition distances,
see [`compute_distances`](@ref).

The `data` contains burst simulation samples for `n_anchor` anchor points,
and can be in one of two layouts `L`.

In the `L=Contiguous` layout, the `data` is an `Array{T,3}`, i.e.,
the number of samples `n_samples` is equal for each anchor,
and the `data` has the shape `(d, n_samples, n_anchors)`.

In the `L=Jagged` layout, the `data` is an `Vector{Array{T,2}}`, i.e.,
each anchor may have a different number of samples,
it is `length(data) = n_anchors`, and `size(data[i]) = (d, n_samples_i)`.

Optionally, `weights` can be specified that give each sample in `data` a weight.
Thus, the layout and shape of `weights` has to match the `data`:
In the `L=Contiguous` layout, the `weights` are an `Array{W,2}` of shape `(n_samples, n_anchors)`,
and in the `L=Jagged` layout, the `weights` are a `Vector{Vector{W}}` of length `n_anchors`.

The type of data points is `T<:Real`, and the type of the weights is `W<:Real`
if weights were provided and `W=Nothing` otherwise.
"""
struct TransitionDistanceProblem{T<:Real,W<:Union{Real,Nothing},L<:AbstractDataLayout}
    data::Union{ContiguousData{T},JaggedData{T}}
    weights::Union{ContiguousWeights{W},JaggedWeights{W},Nothing}

    function TransitionDistanceProblem(
        data::Union{ContiguousData{T},JaggedData{T}},
        weights::Union{ContiguousWeights{W},JaggedWeights{W},Nothing},
    ) where {T<:Real,W<:Real}
        L = data isa ContiguousData ? Contiguous : Jagged

        if data isa JaggedData
            # check that all `d` are the same
            d = size(data[1], 1)
            for x in data
                size(x, 1) == d ||
                    throw(ArgumentError("dimension `d` in data must not change"))
            end
        end

        if isnothing(weights)
            return new{T,Nothing,L}(data, weights)
        end

        if (weights isa ContiguousWeights) != (data isa ContiguousData)
            throw(ArgumentError("mixed layouts of data and weights"))
        end

        if L === Contiguous
            size(data)[2:3] == size(weights) ||
                throw(ArgumentError("shapes of data and weights do not match"))
            return new{T,W,L}(data, weights)
        end

        # L === Jagged
        # check same n_anchors
        length(data) == length(weights) ||
            throw(ArgumentError("lengths of data and weights do not match"))

        # check same sample sizes
        for (x, y) in zip(data, weights)
            size(x, 2) == length(y) ||
                throw(ArgumentError("shapes of data and weights do not match"))
        end
        return new{T,W,L}(data, weights)
    end
end

TransitionDistanceProblem(data; weights=nothing) = TransitionDistanceProblem(data, weights)

"""
    layout(prob::TransitionDistanceProblem)

Layout of `prob`; either [`Contiguous`](@ref) or [`Jagged`](@ref).
"""
layout(::TransitionDistanceProblem{T,W,L}) where {T,W,L} = L

function cat_anchors(
    probs::TransitionDistanceProblem{T,W,Contiguous}...
)::TransitionDistanceProblem{T,W,Contiguous} where {T,W}
    (d, n_samples) = size(probs[1].data)[1:2]
    all(map(p -> size(p.data, 1) == d, probs)) ||
        throw(ArgumentError("dimension `d` of all problems must match"))
    all(map(p -> size(p.data, 2) == n_samples, probs)) ||
        throw(ArgumentError("`n_samples` of all problems must match"))

    data = cat(map(p -> p.data, probs)...; dims=3)
    weights = (W === Nothing) ? nothing : cat(map(p -> p.weights, probs)...; dims=2)
    return TransitionDistanceProblem(data, weights)
end

function append_anchors!(
    prob::TransitionDistanceProblem{T,W,Jagged},
    probs::TransitionDistanceProblem{T,W,Jagged}...,
)::TransitionDistanceProblem{T,W,Jagged} where {T,W}
    d = size(prob.data[1], 1)
    all(map(p -> size(p.data[1], 1) == d, probs)) ||
        throw(ArgumentError("dimensions `d` of all problems must match"))

    append!(prob.data, map(p -> p.data, probs)...)
    if !(W === Nothing)
        append!(prob.weights, map(p -> p.weights, probs)...)
    end
    return prob
end

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
    compute_distances(prob::TransitionDistanceProblem, alg::AbstractTransitionDistanceAlgorithm; progress=false) -> TransitionDistanceResult

Compute pairwise distances of transition density functions, returning a [`TransitionDistanceResult`](@ref) object `res`.

The distance between transition densities ``p_x`` and ``p_y`` for each pair of anchor points ``x`` and ``y``
is estimated using the specified algorithm `alg`,
using the data in the given [`TransitionDistanceProblem`](@ref) `prob`.

The result `res` contains the pairwise distance matrix `res.distances`
and the `res.info` dictionary, which is used to store further information,
see the documentation for each specific algorithm.
"""
function compute_distances(
    prob::TransitionDistanceProblem{T,W,L},
    alg::AbstractTransitionDistanceAlgorithm;
    progress::Bool=false,
)::TransitionDistanceResult where {T,W,L}
    if W === Nothing
        error(
            "No implementation of `compute_distances` for algorithm `$(typeof(alg))` and data layout `$L`",
        )
    end
    error(
        "No implementation of `compute_distances` for algorithm `$(typeof(alg))`, data layout `$L`, and weighted samples",
    )
end

"""
    compute_distances(data, alg::AbstractTransitionDistanceAlgorithm; weights=nothing, progress=false) -> TransitionDistanceResult

The `data`, and optionally `weights`, can be provided directly instead of a [`TransitionDistanceProblem`](@ref).

The `data` contains burst simulation samples for `n_anchor` anchor points,
and can be in one of two layouts.

In the [`Contiguous`](@ref) layout, the `data` is an `Array{T,3}`, i.e.,
the number of samples `n_samples` is equal for each anchor,
and the `data` has the shape `(d, n_samples, n_anchors)`.

In the [`Jagged`](@ref) layout, the `data` is an `Vector{Array{T,2}}`, i.e.,
each anchor may have a different number of samples,
it is `length(data) = n_anchors`, and `size(data[i]) = (d, n_samples_i)`.

The `weights` give each sample in `data` a weight.
Thus, the layout and shape of `weights` has to match the `data`:
In the `L=Contiguous` layout, the `weights` are an `Array{W,2}` of shape `(n_samples, n_anchors)`,
and in the `L=Jagged` layout, the `weights` are a `Vector{Vector{W}}` of length `n_anchors`.
"""
function compute_distances(
    data::Union{ContiguousData{T},JaggedData{T}},
    alg::AbstractTransitionDistanceAlgorithm;
    weights::Union{ContiguousWeights{W},JaggedWeights{W},Nothing}=nothing,
    kwargs...,
)::TransitionDistanceResult where {T<:Real,W<:Real}
    prob = TransitionDistanceProblem(data, weights)
    compute_distances(prob, alg; kwargs...)
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
    compute_transition_manifold(prob::TransitionDistanceProblem, distance_alg::AbstractTransitionDistanceAlgorithm, embedding_alg::AbstractEmbeddingAlgorithm; n_coordinates, progress::Bool=false) -> Tuple{TransitionDistanceResult,EmbeddingResult}

Compute an embedding using the `embedding_alg` from the transition distances that are calculated using the `distance_alg`.

This is an auxiliary function that simply first calls [`compute_distances`](@ref) and then [`compute_embedding`](@ref).
"""
function compute_transition_manifold(
    prob::TransitionDistanceProblem,
    distance_alg::AbstractTransitionDistanceAlgorithm,
    embedding_alg::AbstractEmbeddingAlgorithm;
    n_coordinates::Int=typemax(Int),
    progress::Bool=false,
)::Tuple{TransitionDistanceResult,EmbeddingResult}
    dres = compute_distances(prob, distance_alg; progress=progress)
    eres = compute_embedding(
        dres, embedding_alg; n_coordinates=n_coordinates, progress=progress
    )
    return (dres, eres)
end

"""
    compute_transition_manifold(data, distance_alg::AbstractTransitionDistanceAlgorithm, embedding_alg::AbstractEmbeddingAlgorithm; weights=nothing, kwargs...) -> Tuple{TransitionDistanceResult,EmbeddingResult}

The `data`, and optionally `weights`, can be provided directly instead of a [`TransitionDistanceProblem`](@ref), see also [`compute_distances`](@ref).
"""
function compute_transition_manifold(
    data::Union{ContiguousData{T},JaggedData{T}},
    distance_alg::AbstractTransitionDistanceAlgorithm,
    embedding_alg::AbstractEmbeddingAlgorithm;
    weights::Union{ContiguousWeights{W},JaggedWeights{W},Nothing}=nothing,
    n_coordinates::Int=typemax(Int),
    progress::Bool=false,
)::Tuple{TransitionDistanceResult,EmbeddingResult} where {T<:Real,W<:Real}
    dres = compute_distances(data, distance_alg; weights=weights, progress=progress)
    eres = compute_embedding(
        dres, embedding_alg; n_coordinates=n_coordinates, progress=progress
    )
    return (dres, eres)
end

"""
    PreprocessResult

Struct for storing the result of [`preprocess`](@ref).
"""
struct PreprocessResult
    prob::TransitionDistanceProblem
    info::Dict{String,<:Any}
end

"""
    preprocess(data) -> PreprocessResult

Preprocess `data` so that it can be used in [`compute_distances`](@ref), returning a [`PreprocessResult`](@ref) object `res`.

The result `res` contains a [`TransitionDistanceProblem`](@ref) at `res.prob`, and the `res.info` dictionary,
which is used to store further information.

See the methods below associated to different data types.
"""
function preprocess(data)::PreprocessResult
    error("No implementation of `preprocess` for data of type `$(typeof(data))`")
end

"""
    compute_distances(pres::PreprocessResult, alg::AbstractTransitionDistanceAlgorithm; kwargs...) -> TransitionDistanceResult

A [`PreprocessResult`](@ref) `pres` can be provided directly instead of a [`TransitionDistanceProblem`](@ref).
"""
compute_distances(
    pres::PreprocessResult, alg::AbstractTransitionDistanceAlgorithm; kwargs...
) = compute_distances(pres.prob, alg; kwargs...)

"""
    compute_transition_manifold(pres::PreprocessResult, distance_alg::AbstractTransitionDistanceAlgorithm, embedding_alg::AbstractEmbeddingAlgorithm; kwargs...) -> Tuple{TransitionDistanceResult,EmbeddingResult}

A [`PreprocessResult`](@ref) `pres` can be provided directly instead of a [`TransitionDistanceProblem`](@ref).
"""
compute_transition_manifold(
    pres::PreprocessResult,
    distance_alg::AbstractTransitionDistanceAlgorithm,
    embedding_alg::AbstractEmbeddingAlgorithm;
    kwargs...,
) = compute_transition_manifold(pres.prob, distance_alg, embedding_alg; kwargs...)
