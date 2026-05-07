module KernelMMDExt

using TransitionManifolds
using KernelFunctions
using Distances: SemiMetric
using ProgressMeter: Progress, next!
using LinearAlgebra: BLAS
using StatsBase: mean

# ---------------- KernelDStatMMD ---------------------------
"""
    compute_distances(data, alg::KernelDStatMMD; kwargs...) -> TransitionDistanceResult

When using the [`KernelDStatMMD`](@ref) algorithm, the `res.info` dictionary contains

  - `res.info["elapsed"]`: the elapsed time
"""
function TransitionManifolds.compute_distances(
    data::AbstractArray{T,3}, alg::KernelDStatMMD{<:Kernel}; progress::Bool=false
)::TransitionDistanceResult{T} where {T<:AbstractFloat}
    !isa(alg.kernel.kernel.metric, SemiMetric) && @warn "The metric is not symmetric."
    t1 = @elapsed D = compute_kernel_matrix(data, alg; progress=progress)
    t2 = @elapsed TransitionManifolds.convert_kernel_to_distance_matrix!(D)
    return TransitionDistanceResult(D, Dict("elapsed" => t1 + t2))
end

# This implementation casts integers to Float32. Floats are handled above.
function TransitionManifolds.compute_distances(
    data::AbstractArray{T,3}, alg::KernelDStatMMD{<:Kernel}; kwargs...
)::TransitionDistanceResult where {T<:Real}
    @info "Casting data from $T to Float32 for distance computation"
    return compute_distances(Float32.(data), alg; kwargs...)
end

# Compute the matrix K with K_ij := E[k(x[i], x[j])].
# Since K is symmetric, the entries below the diagonal
# are not filled in and left to be 0.
function compute_kernel_matrix(
    data::AbstractArray{T,3}, alg::KernelDStatMMD{<:Kernel}; progress::Bool=false
)::Matrix{T} where {T<:AbstractFloat}
    n = size(data, 3)
    K = zeros(T, n, n)
    pbar = Progress(
        binomial(n, 2) + n;
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
    x::AbstractMatrix{T}, y::AbstractMatrix{T}, alg::KernelDStatMMD{<:Kernel}
)::T where {T<:AbstractFloat}
    n = min(size(x, 2), size(y, 2))
    out = zero(T)
    k = alg.kernel

    for i in 1:n
        @views out += k(x[:, i], y[:, i])
    end

    return out / n
end

# Estimate E[k(X, X')] from samples x.
# x has shape (d, n).
function kernel_eval(
    x::AbstractMatrix{T}, alg::KernelDStatMMD{<:Kernel}
)::T where {T<:AbstractFloat}
    n = size(x, 2)
    out = zero(T)
    k = alg.kernel

    for i in 1:(n - 1)
        @views out += k(x[:, i], x[:, i + 1])
    end

    return out / (n - 1)
end

# ---------------- KernelVStatMMD ---------------------------
"""
    compute_distances(data, alg::KernelVStatMMD; kwargs...) -> TransitionDistanceResult

When using the [`KernelVStatMMD`](@ref) algorithm, the `res.info` dictionary contains

  - `res.info["elapsed"]`: the elapsed time
"""
function TransitionManifolds.compute_distances(
    data::AbstractArray{T,3}, alg::KernelVStatMMD{<:Kernel}; progress::Bool=false
)::TransitionDistanceResult{T} where {T<:AbstractFloat}
    !isa(alg.kernel.kernel.metric, SemiMetric) && @warn "The metric is not symmetric."
    t1 = @elapsed D = compute_kernel_matrix(data, alg; progress=progress)
    t2 = @elapsed TransitionManifolds.convert_kernel_to_distance_matrix!(D)
    return TransitionDistanceResult(D, Dict("elapsed" => t1 + t2))
end

# This implementation casts integers to Float32. Floats are handled above.
function TransitionManifolds.compute_distances(
    data::AbstractArray{T,3}, alg::KernelVStatMMD{<:Kernel}; kwargs...
)::TransitionDistanceResult where {T<:Real}
    @info "Casting data from $T to Float32 for distance computation"
    return compute_distances(Float32.(data), alg; kwargs...)
end

# Compute the matrix K with K_ij := E[k(x[i], x[j])].
# Since K is symmetric, the entries below the diagonal
# are not filled in and left to be 0.
function compute_kernel_matrix(
    data::AbstractArray{T,3}, alg::KernelVStatMMD{<:Kernel}; progress::Bool=false
)::Matrix{T} where {T<:AbstractFloat}
    n = size(data, 3)
    K = zeros(T, n, n)
    pbar = Progress(
        binomial(n, 2) + n;
        enabled=progress,
        showspeed=true,
        desc="Computing Distance Matrix:",
    )

    # limit BLAS to one thread
    # (some pairwise distance computations are matrix multiplications,
    # which BLAS would parallelize)
    blas_threads_before = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    Threads.@threads for i in axes(data, 3)
        buffer = zeros(T, size(data, 2), size(data, 2))
        for j in 1:i
            @views K[j, i] = kernel_eval(data[:, :, j], data[:, :, i], buffer, alg)
        end
        next!(pbar; step=i, showvalues=[("Iter", "$(pbar.counter) / $(pbar.n)")])
    end

    # restore the number of BLAS threads
    BLAS.set_num_threads(blas_threads_before)

    return K
end

# Estimate E[k(X, Y)] from samples x and y.
# x has shape (d, n) and y has shape (d, m).
function kernel_eval(
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    buffer::AbstractMatrix{T},
    alg::KernelVStatMMD{<:Kernel},
)::T where {T<:AbstractFloat}
    kernelmatrix!(buffer, alg.kernel, ColVecs(x), ColVecs(y))
    mean(buffer)
end

end # module
