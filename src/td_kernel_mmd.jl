"""
    KernelDStatMMD(kernel::Kernel) <: AbstractTransitionDistanceAlgorithm

Struct for using the maximum mean discrepancy (MMD) with any `kernel` from `KernelFunctions`
and D-Statistic estimation to compute the transition density distances.

The MMD between random variables ``X`` and ``Y`` is given by

```math
E[k(X, X')] + E[k(Y, Y')] - 2 E[k(X, Y)],
```

where ``k`` is the `kernel`.

The D-Statistic is used to estimate the above expected values from samples.
The computational cost is linear in the number of samples.
For a more accurate estimator that scales quadratically in the number of samples,
see [`KernelVStatMMD`](@ref).
"""
struct KernelDStatMMD{T} <: AbstractTransitionDistanceAlgorithm
    kernel::T
end

function compute_distances(::TransitionDistanceProblem, ::KernelDStatMMD; kwargs...)
    error(
        "Import the `KernelFunctions` package and provide a `kernel <: Kernel`: `KernelDStatMMD(kernel)`",
    )
end

"""
    KernelVStatMMD(kernel::Kernel) <: AbstractTransitionDistanceAlgorithm

Struct for using the maximum mean discrepancy (MMD) with any `kernel` from `KernelFunctions`
and V-Statistic estimation to compute the transition density distances.

The MMD between random variables ``X`` and ``Y`` is given by

```math
E[k(X, X')] + E[k(Y, Y')] - 2 E[k(X, Y)],
```

where ``k`` is the `kernel`.

The V-Statistic is used to estimate the above expected values from samples.
The computational cost is quadratic in the number of samples.
For a less costly and less accurate estimator see [`KernelDStatMMD`](@ref).
"""
struct KernelVStatMMD{T} <: AbstractTransitionDistanceAlgorithm
    kernel::T
end

function compute_distances(::TransitionDistanceProblem, ::KernelVStatMMD; kwargs...)
    error(
        "Import the `KernelFunctions` package and provide a `kernel <: Kernel`: `KernelVStatMMD(kernel)`",
    )
end

"""
    KernelUStatMMD(kernel::Kernel) <: AbstractTransitionDistanceAlgorithm

Struct for using the maximum mean discrepancy (MMD) with any `kernel` from `KernelFunctions`
and U-Statistic estimation to compute the transition density distances.

The MMD between random variables ``X`` and ``Y`` is given by

```math
E[k(X, X')] + E[k(Y, Y')] - 2 E[k(X, Y)],
```

where ``k`` is the `kernel`.

The U-Statistic estimates ``E[k(X, X')]`` and ``E[k(Y, Y')]`` using only the distinct sample pairs,
excluding the self-pairs ``k(x_i, x_i)``. This gives a less biased estimator than [`KernelVStatMMD`](@ref)
at the same computational cost, assuming the `kernel` is normalized such that ``k(x, x) = 1`` for all ``x``
(e.g. a Gaussian/RBF kernel). The cross term ``E[k(X, Y)]`` is estimated the same way as in [`KernelVStatMMD`](@ref).
"""
struct KernelUStatMMD{T} <: AbstractTransitionDistanceAlgorithm
    kernel::T
end

function compute_distances(::TransitionDistanceProblem, ::KernelUStatMMD; kwargs...)
    error(
        "Import the `KernelFunctions` package and provide a `kernel <: Kernel`: `KernelUStatMMD(kernel)`",
    )
end
