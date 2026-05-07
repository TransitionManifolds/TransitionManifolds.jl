# ---------------- GaussianVStatMMD ---------------------------
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

function compute_distances(
    data::AbstractArray{<:Real,3}, alg::GaussianVStatMMD; progress::Bool=false
)::TransitionDistanceResult
    # TODO: implement!

    error("Not implemented yet!")
end

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
    compute_distances(data, alg::GaussianDStatMMD; kwargs...) -> TransitionDistanceResult

When using the [`GaussianDStatMMD`](@ref) algorithm, the `res.info` dictionary contains

  - `res.info["elapsed"]`: the elapsed time
  - `res.info["bandwidth"]`: the used bandwidth
"""
function compute_distances(
    data::AbstractArray{T,3}, alg::GaussianDStatMMD; progress::Bool=false
)::TransitionDistanceResult{T} where {T<:AbstractFloat}
    # automatic bandwidth selection
    if isnothing(alg.bandwidth)
        n_sub_sample = min(size(data, 2) * size(data, 3), 100) # 100 random points if possible
        subset = stack(sample(eachslice(data; dims=(2, 3)), n_sub_sample; replace=false))
        bandwidth = tune_bandwidth_gaussian(subset)
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
    data::AbstractArray{T,3}, alg::GaussianDStatMMD; kwargs...
)::TransitionDistanceResult where {T<:Real}
    @info "Casting data from $T to Float32 for distance computation"
    return compute_distances(Float32.(data), alg; kwargs...)
end

# Compute the matrix K with K_ij := E[k(x[i], x[j])].
# Since K is symmetric, the entries below the diagonal
# are not filled in and left to be 0.
function compute_kernel_matrix(
    data::AbstractArray{T,3}, alg::GaussianDStatMMD; progress::Bool=false
)::Matrix{T} where {T<:AbstractFloat}
    n = size(data, 3)
    K = zeros(T, n, n)
    pbar = Progress(
        binomial(n, 2) + 1;
        enabled=progress,
        showspeed=true,
        desc="Computing Distance Matrix:",
    )

    Threads.@threads for i in axes(data, 3)
        @views K[i, i] = kernel_eval(data[:, :, i], alg)
    end
    next!(pbar; step=n, showvalues=[("Iter", "$(pbar.counter) / $(pbar.n)")])

    Threads.@threads for i in axes(data, 3)
        for j in 1:(i - 1)
            @views K[j, i] = kernel_eval(data[:, :, j], data[:, :, i], alg)
        end
        next!(pbar; step=(i - 1), showvalues=[("Iter", "$(pbar.counter) / $(pbar.n)")])
    end

    return K
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
