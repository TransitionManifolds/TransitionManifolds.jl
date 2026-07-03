"""
    compute_kernel_matrix(data::JaggedData{T}, alg::AbstractTransitionDistanceAlgorithm; progress=false) -> Matrix{T}

Compute the kernel matrix ``K`` with ``K_{ij} := E[k(x[i], x[j])]``.

Since K is symmetric, the entries below the diagonal are not filled in and left to be 0.

For the given `alg`, the methods `kernel_eval(x, alg)` and `kernel_eval(x, y, alg)` have to be implemented.
"""
function compute_kernel_matrix(
    data::JaggedData{T}, alg::AbstractTransitionDistanceAlgorithm; progress::Bool=false
)::Matrix{T} where {T<:AbstractFloat}
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

    Threads.@threads :greedy for i in eachindex(data)
        for j in 1:(i - 1)
            K[j, i] = kernel_eval(data[j], data[i], alg)
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
