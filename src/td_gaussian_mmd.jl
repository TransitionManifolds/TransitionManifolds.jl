# ---------------- GaussianDStatMMD ---------------------------

"""
    GaussianDStatMMD(bandwidth=nothing) <: AbstractTransitionDistanceAlgorithm

Struct for using the maximum mean discrepancy (MMD) with a Gaussian kernel
and D-Statistic estimation to compute the transition density distances.

The MMD between random variables ``X`` and ``Y`` is given by

```math
E[k(X, X')] + E[k(Y, Y')] - 2 E[k(X, Y)].
```

Here, ``k`` is a Gaussian kernel:
``k(x, y) := exp(-||x - y||^2 / σ^2)``
where ``σ`` is called the `bandwidth`.
The D-Statistic is used to estimate the above expected values from samples.
The computational cost is linear in the number of samples.
For a more accurate estimator that scales quadratically in the number of samples,
see [`GaussianVStatMMD`](@ref).

The `bandwidth` is either a number > 0 or `nothing`, in which case a reasonable bandwidth is chosen automatically based on the samples.
"""
struct GaussianDStatMMD <: AbstractTransitionDistanceAlgorithm
    bandwidth::Union{Float64,Nothing}

    function GaussianDStatMMD(bandwidth::Union{Real,Nothing})
        isnothing(bandwidth) ||
            bandwidth > 0 ||
            throw(ArgumentError("`bandwidth` has to be > 0"))
        return new(bandwidth)
    end
end
GaussianDStatMMD(; bandwidth=nothing) = GaussianDStatMMD(bandwidth)

"""
    compute_distances(prob, alg::GaussianDStatMMD; kwargs...) -> TransitionDistanceResult

The [`GaussianDStatMMD`](@ref) algorithm works with [`Contiguous`](@ref) and [`Jagged`](@ref) layout. Weighted samples are not supported.
The `res.info` dictionary contains

  - `res.info["elapsed"]`: the elapsed time
  - `res.info["bandwidth"]`: the used bandwidth
"""
function compute_distances(
    prob::TransitionDistanceProblem{T,Nothing,Jagged},
    alg::GaussianDStatMMD;
    progress::Bool=false,
)::TransitionDistanceResult{T} where {T<:AbstractFloat}
    data = prob.data

    # automatic bandwidth selection
    if isnothing(alg.bandwidth)
        subsamples = subsamples_from_jagged(data, 100)
        bandwidth = tune_bandwidth_gaussian(subsamples)
        alg = GaussianDStatMMD(bandwidth)
    end

    t1 = @elapsed D = compute_kernel_matrix(data, alg; progress=progress)
    t2 = @elapsed convert_kernel_to_distance_matrix!(D)
    return TransitionDistanceResult(
        D, Dict("bandwidth" => alg.bandwidth, "elapsed" => t1 + t2)
    )
end

# This implementation casts integers to Float32. Floats are handled above.
function compute_distances(
    prob::TransitionDistanceProblem{T,Nothing,Jagged}, alg::GaussianDStatMMD; kwargs...
)::TransitionDistanceResult where {T<:Real}
    @info "Casting data from $T to Float32 for distance computation"
    prob = TransitionDistanceProblem(map(x -> Float32.(x), prob.data))
    return compute_distances(prob, alg; kwargs...)
end

# This implementation converts Contiguous to Jagged layout. Jagged is handled above.
function compute_distances(
    prob::TransitionDistanceProblem{T,Nothing,Contiguous}, alg::GaussianDStatMMD; kwargs...
)::TransitionDistanceResult where {T<:Real}
    return compute_distances(convert_contiguous_to_jagged(prob), alg; kwargs...)
end

# Estimate E[k(X, Y)] from samples x and y.
# x has shape (d, n) and y has shape (d, m).
function kernel_eval(
    x::AbstractMatrix{T}, y::AbstractMatrix{T}, alg::GaussianDStatMMD
)::T where {T<:AbstractFloat}
    n = min(size(x, 2), size(y, 2))
    d = min(size(x, 1), size(y, 1))
    inv_sigma_sq = T(-1.0 / (alg.bandwidth * alg.bandwidth))
    out = zero(T)

    for i in 1:n
        dist_sq = zero(T)
        @turbo for k in 1:d
            diff = x[k, i] - y[k, i]
            dist_sq = muladd(diff, diff, dist_sq)  # fused multiply-add
        end
        out += exp(dist_sq * inv_sigma_sq)
    end

    return out / n
end

# Estimate E[k(X, X')] from samples x.
# x has shape (d, n).
function kernel_eval(
    x::AbstractMatrix{T}, alg::GaussianDStatMMD
)::T where {T<:AbstractFloat}
    n = size(x, 2)
    if n == 1 # otherwise we divide by 0 later
        # returning "1" so that all anchors have a large distance to this one
        return one(T)
    end

    inv_sigma_sq = T(-1.0 / (alg.bandwidth * alg.bandwidth))
    out = zero(T)

    for i in 1:(n - 1)
        dist_sq = zero(T)
        @turbo for k in axes(x, 1)
            # use off-diagonal entries since the diagonal would just yield `1`
            diff = x[k, i] - x[k, i + 1]
            dist_sq = muladd(diff, diff, dist_sq)  # fused multiply-add
        end
        out += exp(dist_sq * inv_sigma_sq)
    end
    return out / (n - 1)
end

# ---------------- GaussianVStatMMD ---------------------------
# TODO: docstring
"""
    GaussianVStatMMD(...) <: AbstractTransitionDistanceAlgorithm

Not implemented yet!
"""
struct GaussianVStatMMD <: AbstractTransitionDistanceAlgorithm
    bandwidth::Union{Float64,Nothing}
    blocksize::Int

    function GaussianVStatMMD(bandwidth::Union{Real,Nothing}, blocksize::Integer)
        isnothing(bandwidth) ||
            bandwidth > 0 ||
            throw(ArgumentError("bandwidth has to be > 0"))
        blocksize > 0 || throw(ArgumentError("blocksize has to be > 0"))
        new(bandwidth, blocksize)
    end
end
GaussianVStatMMD(; bandwidth::Union{Real,Nothing}=nothing, blocksize::Integer=20) =
    GaussianVStatMMD(bandwidth, blocksize)

# TODO: docstring
function compute_distances(
    prob::TransitionDistanceProblem{T,Nothing,Jagged},
    alg::GaussianVStatMMD;
    progress::Bool=false,
)::TransitionDistanceResult where {T<:AbstractFloat}
    data = prob.data

    # automatic bandwidth selection
    if isnothing(alg.bandwidth)
        subsamples = subsamples_from_jagged(data, 100)
        bandwidth = tune_bandwidth_gaussian(subsamples)
        alg = GaussianVStatMMD(bandwidth, alg.blocksize)
    end

    t1 = @elapsed D = compute_kernel_matrix(data, alg; progress=progress)
    t2 = @elapsed convert_kernel_to_distance_matrix!(D)
    return TransitionDistanceResult(
        D, Dict("bandwidth" => alg.bandwidth, "elapsed" => t1 + t2)
    )
end

# TODO: implementation for Contiguous using blocks

# TODO: casting

# Estimate E[k(X, Y)] from samples x and y.
# x has shape (d, n) and y has shape (d, m).
function kernel_eval(
    x::AbstractMatrix{T}, y::AbstractMatrix{T}, alg::GaussianVStatMMD
)::T where {T<:AbstractFloat}
    inv_sigma_sq = T(-1.0 / (alg.bandwidth * alg.bandwidth))
    dist_sq = pairwise(SqEuclidean(), x, y)
    dist_sq .= exp.(inv_sigma_sq .* dist_sq)
    return mean(dist_sq)
end

function kernel_eval(
    x::AbstractMatrix{T}, alg::GaussianVStatMMD
)::T where {T<:AbstractFloat}
    kernel_eval(x, x, alg)
end
