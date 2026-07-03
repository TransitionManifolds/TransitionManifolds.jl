# Convert a TransitionDistanceProblem from contiguous to jagged layout using slices.
function convert_contiguous_to_jagged(
    prob::TransitionDistanceProblem{T,W,Contiguous}
)::TransitionDistanceProblem{T,W,Jagged} where {T,W}
    data = eachslice(prob.data; dims=3)
    weights = (W === Nothing) ? nothing : eachslice(prob.weights; dims=2)
    return TransitionDistanceProblem(data, weights)
end

"""
    subsamples_from_data(data, n_samples) -> Matrix

Generate `n_samples` random samples from jagged or contiguous `data`.

The output is a `(d, n_samples)` matrix.
If there are less than `n_samples` data points in `data`, the output will
contain only all data points, i.e., will have less than `n_samples` columns.
"""
function subsamples_from_data(
    data::JaggedData{T}, n_samples::Int
)::AbstractArray{T,2} where {T}
    # create a vector of column views containing all data points
    all_cols = mapreduce(eachcol, vcat, data)

    n_samples = min(n_samples, length(all_cols))
    return stack(sample(all_cols, n_samples; replace=false))
end

function subsamples_from_data(
    data::ContiguousData{T}, n_samples::Int
)::AbstractArray{T,2} where {T}
    all_slices = eachslice(data; dims=(2, 3))
    n_samples = min(n_samples, length(all_slices))
    return stack(sample(all_slices, n_samples; replace=false))
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
