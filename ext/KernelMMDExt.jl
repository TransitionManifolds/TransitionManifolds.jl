module KernelMMDExt

using TransitionManifolds
using KernelFunctions
using Distances: SemiMetric
using ProgressMeter: Progress, next!
using LinearAlgebra: BLAS
using StatsBase: mean

# ---------------- KernelDStatMMD ---------------------------
"""
    compute_distances(prob, alg::KernelDStatMMD; kwargs...) -> TransitionDistanceResult

The [`KernelDStatMMD`](@ref) algorithm works with [`Contiguous`](@ref) and [`Jagged`](@ref) layout. Weighted samples are not supported.
The `res.info` dictionary contains

  - `res.info["elapsed"]`: the elapsed time
"""
function TransitionManifolds.compute_distances(
    prob::TransitionDistanceProblem{T,Nothing,Jagged},
    alg::KernelDStatMMD{<:Kernel};
    progress::Bool=false,
)::TransitionDistanceResult{T} where {T<:AbstractFloat}
    !isa(alg.kernel.kernel.metric, SemiMetric) && @warn "The metric is not symmetric."
    t1 = @elapsed D = TransitionManifolds.compute_kernel_matrix(
        prob.data, alg; progress=progress
    )
    t2 = @elapsed TransitionManifolds.convert_kernel_to_distance_matrix!(D)
    return TransitionDistanceResult(D, Dict("elapsed" => t1 + t2))
end

# This implementation casts integers to Float32. Floats are handled above.
function TransitionManifolds.compute_distances(
    prob::TransitionDistanceProblem{T,Nothing,Jagged},
    alg::KernelDStatMMD{<:Kernel};
    kwargs...,
)::TransitionDistanceResult where {T<:Real}
    @info "Casting data from $T to Float32 for distance computation"
    prob = TransitionDistanceProblem(map(x -> Float32.(x), prob.data))
    return compute_distances(prob, alg; kwargs...)
end

# This implementation converts Contiguous to Jagged layout. Jagged is handled above.
function TransitionManifolds.compute_distances(
    prob::TransitionDistanceProblem{T,Nothing,Contiguous},
    alg::KernelDStatMMD{<:Kernel};
    kwargs...,
)::TransitionDistanceResult where {T<:Real}
    return compute_distances(
        TransitionManifolds.convert_contiguous_to_jagged(prob), alg; kwargs...
    )
end

function TransitionManifolds.kernel_eval(
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

function TransitionManifolds.kernel_eval(
    x::AbstractMatrix{T}, alg::KernelDStatMMD{<:Kernel}
)::T where {T<:AbstractFloat}
    n = size(x, 2)
    if n == 1 # otherwise we divide by 0 later
        # returning "1" so that all anchors have a large distance to this one
        return one(T)
    end

    out = zero(T)
    k = alg.kernel

    for i in 1:(n - 1)
        @views out += k(x[:, i], x[:, i + 1])
    end

    return out / (n - 1)
end

# ---------------- KernelVStatMMD ---------------------------
"""
    compute_distances(prob, alg::KernelVStatMMD; kwargs...) -> TransitionDistanceResult

The [`KernelVStatMMD`](@ref) algorithm works with [`Contiguous`](@ref) and [`Jagged`](@ref) layout. Weighted samples are not supported.
The `res.info` dictionary contains

  - `res.info["elapsed"]`: the elapsed time
"""
function TransitionManifolds.compute_distances(
    prob::TransitionDistanceProblem{T,Nothing,Jagged},
    alg::KernelVStatMMD{<:Kernel};
    progress::Bool=false,
)::TransitionDistanceResult{T} where {T<:AbstractFloat}
    !isa(alg.kernel.kernel.metric, SemiMetric) && @warn "The metric is not symmetric."

    t1 = @elapsed D = TransitionManifolds.compute_kernel_matrix_buffered(
        prob.data, alg; progress=progress
    )
    t2 = @elapsed TransitionManifolds.convert_kernel_to_distance_matrix!(D)
    return TransitionDistanceResult(D, Dict("elapsed" => t1 + t2))
end

# This implementation casts integers to Float32. Floats are handled above.
function TransitionManifolds.compute_distances(
    prob::TransitionDistanceProblem{T,Nothing,Jagged},
    alg::KernelVStatMMD{<:Kernel};
    kwargs...,
)::TransitionDistanceResult where {T<:Real}
    @info "Casting data from $T to Float32 for distance computation"
    prob = TransitionDistanceProblem(map(x -> Float32.(x), prob.data))
    return compute_distances(prob, alg; kwargs...)
end

# This implementation converts Contiguous to Jagged layout. Jagged is handled above.
function TransitionManifolds.compute_distances(
    prob::TransitionDistanceProblem{T,Nothing,Contiguous},
    alg::KernelVStatMMD{<:Kernel};
    kwargs...,
)::TransitionDistanceResult where {T<:Real}
    return compute_distances(
        TransitionManifolds.convert_contiguous_to_jagged(prob), alg; kwargs...
    )
end

function TransitionManifolds.kernel_eval(
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    alg::KernelVStatMMD{<:Kernel},
    buffer::AbstractMatrix{T},
)::T where {T<:AbstractFloat}
    kernelmatrix!(buffer, alg.kernel, ColVecs(x), ColVecs(y))
    return mean(buffer)
end

end # module
