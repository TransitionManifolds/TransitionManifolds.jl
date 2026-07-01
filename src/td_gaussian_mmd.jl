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
        subsamples = subsamples_from_data(data, 100)
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
    prob::TransitionDistanceProblem{T,Nothing,<:AbstractDataLayout},
    alg::GaussianVStatMMD;
    progress::Bool=false,
)::TransitionDistanceResult where {T<:AbstractFloat}
    data = prob.data

    # automatic bandwidth selection
    if isnothing(alg.bandwidth)
        subsamples = subsamples_from_data(data, 100)
        bandwidth = tune_bandwidth_gaussian(subsamples)
        alg = GaussianVStatMMD(bandwidth, alg.blocksize)
    end

    # For jagged layout, `compute_kernel_matrix` uses the standard method from `utils.jl`.
    # For contigous layout, there is a special implementation below.
    t1 = @elapsed D = compute_kernel_matrix(data, alg; progress=progress)

    t2 = @elapsed convert_kernel_to_distance_matrix!(D)
    return TransitionDistanceResult(
        D, Dict("bandwidth" => alg.bandwidth, "elapsed" => t1 + t2)
    )
end

# VStat + Contiguous: blockwise computation 
function compute_kernel_matrix(
    data::ContiguousData{T}, alg::GaussianVStatMMD; progress::Bool=false
)::Matrix{T} where {T<:AbstractFloat}
    dim, n_samples, n_anchors = size(data)
    total_points = n_samples * n_anchors
    inv_sigma_sq = T(-1.0 / (alg.bandwidth * alg.bandwidth))
    K = zeros(T, n_anchors, n_anchors)

    blocksize = min(alg.blocksize, n_anchors)
    while n_anchors % blocksize != 0
        blocksize -= 1
    end

    # We need ||x||^2 for the distance expansion ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    # compute norm once for entire dataset
    x_flat = reshape(data, (dim, total_points))
    norms_sq = sum(abs2, x_flat; dims=1) # (1, total_points)

    # set BLAS threads for manual threading
    blas_threads_before = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    n_iter = div(n_anchors, blocksize)
    pbar = Progress(
        (binomial(n_iter, 2) + n_iter) * blocksize * blocksize;
        enabled=progress,
        showspeed=true,
        desc="Computing Kernel Matrix:",
    )

    Threads.@threads for i_start in 1:blocksize:n_anchors
        i_end = i_start + blocksize - 1

        idx_start_i = (i_start - 1) * n_samples + 1
        idx_end_i = i_end * n_samples

        buffer = zeros(T, n_samples * blocksize, n_samples * blocksize)

        # shape: (dim, blocksize * samples)
        Xi = view(x_flat, :, idx_start_i:idx_end_i)
        norms_i = view(norms_sq, :, idx_start_i:idx_end_i)

        for j_start in i_start:blocksize:n_anchors
            j_end = j_start + blocksize - 1

            idx_start_j = (j_start - 1) * n_samples + 1
            idx_end_j = j_end * n_samples

            Xj = view(x_flat, :, idx_start_j:idx_end_j)
            norms_j = view(norms_sq, :, idx_start_j:idx_end_j)

            # (Samples*blocksize x dim) * (dim x Samples*blocksize) => (Samples*blocksize) x (Samples*blocksize)
            # This consumes the dim in the dot product summation
            mul!(buffer, Xi', Xj)

            # ||x||^2 + ||y||^2 - 2<x,y> and kernel
            @tullio buffer[i, j] = exp(
                (norms_i[1, i] + norms_j[1, j] - T(2) * buffer[i, j]) * inv_sigma_sq
            ) (threads = false)

            # reshape to separate anchors
            this_K = reshape(buffer, (n_samples, blocksize, n_samples, blocksize))

            # sum sub-blocks of size (samples x samples)
            @tullio result_tile[i, j] := this_K[s1, i, s2, j] (threads = false)

            K[i_start:i_end, j_start:j_end] = result_tile
            next!(
                pbar;
                step=length(result_tile),
                showvalues=[("Iter", "$(pbar.counter) / $(pbar.n)")],
            )
        end
    end

    BLAS.set_num_threads(blas_threads_before)
    K ./= n_samples^2
    return K
end

# This implementation casts integers to Float32. Floats are handled above.
function compute_distances(
    prob::TransitionDistanceProblem{T,Nothing,<:AbstractDataLayout}, alg::GaussianVStatMMD; kwargs...
)::TransitionDistanceResult where {T<:Real}
    @info "Casting data from $T to Float32 for distance computation"
    prob = TransitionDistanceProblem(map(x -> Float32.(x), prob.data))
    return compute_distances(prob, alg; kwargs...)
end

# Estimate E[k(X, Y)] from samples x and y.
# x has shape (d, n) and y has shape (d, m).
function kernel_eval(
    x::AbstractMatrix{T}, y::AbstractMatrix{T}, alg::GaussianVStatMMD
)::T where {T<:AbstractFloat}
    inv_sigma_sq = T(-1.0 / (alg.bandwidth * alg.bandwidth))
    dist_sq = pairwise(SqEuclidean(), x, y; dims=2)
    dist_sq .= exp.(inv_sigma_sq .* dist_sq)
    return mean(dist_sq)
end

function kernel_eval(
    x::AbstractMatrix{T}, alg::GaussianVStatMMD
)::T where {T<:AbstractFloat}
    kernel_eval(x, x, alg)
end
