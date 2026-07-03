"""
    kernel_eval(x::AbstractMatrix, y::AbstractMatrix, alg::AbstractTransitionDistanceAlgorithm, [buffer]) -> Real

Estimate ``E[k(X, Y)]`` from samples `x` of the random variable ``X`` and samples `y` of the random variable ``Y``.

The kernel `k` is implicitly given by the `alg`.
`x` contains `n` samples and has shape `(d, n)`.
`y` contains `m` samples and has shape `(d, m)`.

# Interface

## Required methods

The `alg` has to implement either `kernel_eval(x, y, alg)` or `kernel_eval(x, y, alg, buffer)`.
The `buffer` has shape `(n, m)` and can be useful for some algorithms to store pairwise computations between the samples.
Implementing `kernel_eval(x, y, alg, buffer)` automatically yields an implementation of `kernel_eval(x, y, alg)`.

## Optional methods

Implement `kernel_eval(x, alg)` or `kernel_eval(x, alg, buffer)` to handle the special case `x=y`.
Default behavior: calls `kernel_eval(x, x, alg)` or `kernel_eval(x, x alg, buffer)`.

## Provided methods

Implementing `kernel_eval(x, y, alg)` for `alg` allows calling [`compute_kernel_matrix`](@ref);
Implementing `kernel_eval(x, y, alg, buffer)` for `alg` allows calling [`compute_kernel_matrix_buffered`](@ref).
"""
function kernel_eval(
    x::AbstractMatrix,
    y::AbstractMatrix,
    alg::AbstractTransitionDistanceAlgorithm,
    buffer::AbstractMatrix,
)::Real
    error("No implementation of `kernel_eval` for algorithm `$(typeof(alg))`")
end

function kernel_eval(
    x::AbstractMatrix{T}, y::AbstractMatrix{T}, alg::AbstractTransitionDistanceAlgorithm
) where {T}
    buffer = Matrix{T}(undef, size(x, 2), size(y, 2))
    return kernel_eval(x, y, alg, buffer)
end

kernel_eval(
    x::AbstractMatrix, alg::AbstractTransitionDistanceAlgorithm, buffer::AbstractMatrix
) = kernel_eval(x, x, alg, buffer)
kernel_eval(x::AbstractMatrix, alg::AbstractTransitionDistanceAlgorithm) =
    kernel_eval(x, x, alg)

"""
    compute_kernel_matrix(data::JaggedData{T}, alg::AbstractTransitionDistanceAlgorithm; progress=false) -> Matrix{T}

Compute the kernel matrix ``K`` with ``K_{ij} := E[k(x[i], x[j])]``.

Since K is symmetric, the entries below the diagonal are not filled in and left to be 0.

Requires that the method `kernel_eval(x, y, alg)` is implemented.
"""
function compute_kernel_matrix(
    data::JaggedData{T}, alg::AbstractTransitionDistanceAlgorithm; progress::Bool=false
)::Matrix{T} where {T}
    n = length(data)
    K = zeros(T, n, n)
    pbar = Progress(
        binomial(n, 2) + n;
        enabled=progress,
        showspeed=true,
        desc="Computing Kernel Matrix:",
    )

    # set BLAS threads for manual threading
    blas_threads_before = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    Threads.@threads for i in eachindex(data)
        K[i, i] = kernel_eval(data[i], alg)
    end
    next!(pbar; step=n, showvalues=[("Iter", "$(pbar.counter) / $(pbar.n)")])

    Threads.@threads :greedy for i in 2:length(data)
        for j in 1:(i - 1)
            K[j, i] = kernel_eval(data[j], data[i], alg)
        end
        next!(pbar; step=(i - 1), showvalues=[("Iter", "$(pbar.counter) / $(pbar.n)")])
    end

    BLAS.set_num_threads(blas_threads_before)
    return K
end

"""
A version of [`compute_kernel_matrix`](@ref) that makes use of a buffer.

Requires that the method `kernel_eval(x, y, alg, buffer)` is implemented.
"""
function compute_kernel_matrix_buffered(
    data::JaggedData{T}, alg::AbstractTransitionDistanceAlgorithm; progress::Bool=false
)::Matrix{T} where {T}
    n = length(data)
    K = zeros(T, n, n)
    pbar = Progress(
        binomial(n, 2) + n;
        enabled=progress,
        showspeed=true,
        desc="Computing Kernel Matrix:",
    )

    # set BLAS threads for manual threading
    blas_threads_before = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    Threads.@threads for i in eachindex(data)
        K[i, i] = kernel_eval(data[i], alg)
    end
    next!(pbar; step=n, showvalues=[("Iter", "$(pbar.counter) / $(pbar.n)")])

    Threads.@threads :greedy for i in 2:length(data)
        nj_max = maximum([size(data[j], 2) for j in 1:(i - 1)])
        buffer = Matrix{T}(undef, size(data[i], 2), nj_max)
        for j in 1:(i - 1)
            bufview = @view buffer[:, 1:size(data[j], 2)]
            K[j, i] = kernel_eval(data[i], data[j], alg, bufview)
        end
        next!(pbar; step=(i - 1), showvalues=[("Iter", "$(pbar.counter) / $(pbar.n)")])
    end

    BLAS.set_num_threads(blas_threads_before)
    return K
end

"""
    convert_kernel_to_distance_matrix!(K::AbstractMatrix)

Convert a kernel matrix `K` to a distance matrix `D` in place:

```math
D_{i,j} = K_{i,i} + K_{j,j} - 2 K_{i,j}.
```

The values below the diagonal of the `K` are not used in the computation (due to symmetry).
`D` will have all elements filled in and it will be 0 on the diagonal.
Futhermore, the elements of `D` are guaranteed to be nonnegative by truncation.
"""
function convert_kernel_to_distance_matrix!(K::AbstractMatrix{T}) where {T<:Real}
    for j in axes(K, 2)
        for i in 1:(j - 1)
            value = K[i, i] + K[j, j] - 2 * K[i, j]
            value = max(value, zero(T))
            K[i, j] = value
            K[j, i] = value
        end
    end
    K[diagind(K)] .= 0
end
