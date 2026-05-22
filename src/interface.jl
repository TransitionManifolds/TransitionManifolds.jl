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

const ContiguousData{T<:Real} = Array{T,3}
const JaggedData{T<:Real} = Vector{Array{T,2}}

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

In the `L=Jagged` case, the `data` is an `Vector{Array{T,2}}`, i.e.,
each anchor may have a different number of samples,
it is `length(data) = n_anchors`, and `size(data[i]) = (d, n_samples_i)`.

Optionally, `weights` can be specified that give each sample in `data` a weight.
Thus, the layout and shape of `weights` has to exactly match the `data`.

The type of data points is `T<:Real`, and the type of the weights is `W<:Real`
if weights were provided and `W=Nothing` otherwise.
"""
struct TransitionDistanceProblem{T<:Real,W<:Union{Real,Nothing},L<:AbstractDataLayout}
    data::Union{ContiguousData{T},JaggedData{T}}
    weights::Union{ContiguousData{W},JaggedData{W},Nothing}

    function TransitionDistanceProblem(
        data::Union{ContiguousData{T},JaggedData{T}},
        weights::Union{ContiguousData{W},JaggedData{W},Nothing},
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
            return new{T,Nothing,L}(data, nothing)
        end

        if (weights isa ContiguousData) != (data isa ContiguousData)
            throw(ArgumentError("mixed layouts of data and weights"))
        end

        if L === Contiguous
            size(data) == size(weights) ||
                throw(ArgumentError("shapes of data and weights do not match"))
            return new{T,W,L}(data, nothing)
        end

        # L === Jagged
        # check same n_anchors
        length(data) == length(weights) ||
            throw(ArgumentError("lenghts of data and weights do not match"))

        # check same sample sizes
        for (x, y) in zip(data, weights)
            size(x) == size(y) ||
                throw(ArgumentError("shapes of data and weights do not match"))
        end
        return new{T,W,L}(data, nothing)
    end
end

TransitionDistanceProblem(data; weights=nothing) = TransitionDistanceProblem(data, weights)

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

"""
    PreprocessResult{T<:Real}

Struct for storing the result of [`preprocess`](@ref).
"""
struct PreprocessResult{T<:Real}
    data::Array{T,3}
    info::Dict{String,<:Any}
end

"""
    preprocess(data) -> PreprocessResult

Preprocess `data` so that it can be used in [`compute_distances`](@ref), returning a [`PreprocessResult`](@ref) object `res`.

The result `res` contains the preprocessed data at `res.data`, and the `res.info` dictionary,
which is used to store further information.

See the methods below associated to different data types.
"""
function preprocess(data)::PreprocessResult
    error("No implementation of `preprocess` for data of type `$(typeof(data))`")
end

"""
    compute_distances(pres::PreprocessResult, alg::AbstractTransitionDistanceAlgorithm; kwargs...) -> TransitionDistanceResult

A [`PreprocessResult`](@ref) `pres` can be provided instead of the `data` array.
"""
compute_distances(
    pres::PreprocessResult, alg::AbstractTransitionDistanceAlgorithm; kwargs...
) = compute_distances(pres.data, alg; kwargs...)

"""
    compute_transition_manifold(pres::PreprocessResult, distance_alg::AbstractTransitionDistanceAlgorithm, embedding_alg::AbstractEmbeddingAlgorithm; kwargs...) -> Tuple{TransitionDistanceResult,EmbeddingResult}

A [`PreprocessResult`](@ref) `pres` can be provided instead of the `data` array.
"""
compute_transition_manifold(
    pres::PreprocessResult,
    distance_alg::AbstractTransitionDistanceAlgorithm,
    embedding_alg::AbstractEmbeddingAlgorithm;
    kwargs...,
) = compute_transition_manifold(pres.data, distance_alg, embedding_alg; kwargs...)
