# Convert a TransitionDistanceProblem from contiguous to jagged layout using slices.
function convert_contiguous_to_jagged(
    prob::TransitionDistanceProblem{T,W,Contiguous}
)::TransitionDistanceProblem{T,W,Jagged} where {T,W}
    data = eachslice(prob.data; dims=3)
    weights = (W === Nothing) ? nothing : eachslice(prob.weights; dims=3)
    return TransitionDistanceProblem(data, weights)
end

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
        binomial(n, 2) + 1;
        enabled=progress,
        showspeed=true,
        desc="Computing Kernel Matrix:",
    )

    Threads.@threads for i in eachindex(data)
        K[i, i] = kernel_eval(data[i], alg)
    end
    next!(pbar; step=n, showvalues=[("Iter", "$(pbar.counter) / $(pbar.n)")])

    Threads.@threads for i in eachindex(data)
        for j in 1:(i - 1)
            K[j, i] = kernel_eval(data[j], data[i], alg)
        end
        next!(pbar; step=(i - 1), showvalues=[("Iter", "$(pbar.counter) / $(pbar.n)")])
    end

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

"""
    subsamples_from_jagged(data::Vector{Matrix}, n_samples) -> Matrix

Generate `n_samples` random samples from jagged `data`.

The output is a `(d, n_samples)` matrix.
If there are less than `n_samples` data points in `data`, the output will
contain only all data points, i.e., will have less than `n_samples` columns.
"""
function subsamples_from_jagged(
    data::JaggedData{T}, n_samples::Int
)::AbstractArray{T,2} where {T}
    # create a vector of column views containing all data points
    all_cols = mapreduce(eachcol, vcat, data)

    n_samples = min(n_samples, length(all_cols))
    return stack(sample(all_cols, n_samples; replace=false))
end

"""
    tune_bandwidth_gaussian(data::AbstractMatrix{<:Real}; quant=0.95, val_at_quant=0.01)

Return the bandwidth ``σ`` of the gaussian kernel

```math
k(x, y) = exp(-||x - y||^2 / σ^2).
```

tuned to the `data` of shape `(d, n_points)`.

The bandwidth is chosen such that for the `quant`-quantile of pairwise distances in the `data` the kernel value is `val_at_quant`,
i.e., for the default arguments the ``95%``-largest distance will produce the kernel value ``0.01``.
"""
function tune_bandwidth_gaussian(
    data::AbstractMatrix{<:Real}; quant::Real=0.95, val_at_quant::Real=0.01
)::Float64
    0 <= quant <= 1 || throw(ArgumentError("`quant` has to be between 0 and 1"))
    val_at_quant > 0 || throw(ArgumentError("`val_at_quant` has to be between > 0"))
    distances_matrix = pairwise(SqEuclidean(), data; dims=2)
    distances = filter(x -> x > 0.0, LowerTriangular(distances_matrix))
    q = quantile(distances, quant)
    σ = sqrt(-q / log(val_at_quant))
    return σ
end

"""
    normalize_cloud(X::AbstractMatrix{<:Real}; quant=0.99) -> Matrix

Center, rotate, and scale a point-cloud `X` of shape `(n_points, n_coordinates)`
so that inliers (below the `quant`-th radius quantile) fit inside the unit cube.

Returns the normalized point-cloud.
"""
function normalize_cloud(X::AbstractMatrix{<:Real}; quant::Real=0.99)
    0.0 <= quant <= 1.0 || throw(ArgumentError("`quant` has to be between 0 and 1"))

    # centering
    μ = median(X; dims=1)
    Xc = X .- μ

    # remove outliers
    rs = norm.(eachrow(Xc))
    r_threshold = quantile(rs, quant)
    mask = rs .<= r_threshold

    # rotate via PCA
    Xin = @view Xc[mask, :]
    C = cov(Xin; dims=1)
    _, V = eigen(C)
    Xr = Xc * V

    # scale inliers into [-1,1]
    scale = maximum(abs, view(Xr, mask, :))
    Xr .= Xr ./ scale
    reverse!(Xr; dims=2)
    return Xr
end
