"""
    DiffusionMaps(bandwidth=nothing, alpha=0.5) <: AbstractEmbeddingAlgorithm

Struct for using diffusion maps to compute an embedding.

In the diffusion maps algorithm, the similarity matrix ``L`` is computed
from the pairwise distance matrix ``Δ`` via

```math
L_{ij} := exp(-Δ_{i,j}^2 / σ^2)
```

where ``σ`` is called the `bandwidth`.
Then ``L`` is normalized according to the normalization parameter ``α``, i.e.,

```math
D^{-α} L D^{-α}
```

where ``D`` is the diagonal matrix containing the row-sums of ``L``.
The resulting matrix is then row-normalized to obtain the stochastic diffusion matrix ``M``.
Finally, the coordinates of the embedding are given as the dominant eigenvectors of ``M``,
multiplied by the associated eigenvalues.

The `bandwidth` is either a number > 0 or `nothing`, in which case a reasonable bandwidth is chosen automatically.
Typical choices for the normalization parameter `alpha` are

  - `α = 0`: Graph Laplacian normalization
  - `α = 0.5`: Fokker-Planck normalization
  - `α = 1`: Laplace-Beltrami normalization.
"""
struct DiffusionMaps <: AbstractEmbeddingAlgorithm
    bandwidth::Union{Float64,Nothing}
    alpha::Float64

    function DiffusionMaps(bandwidth::Union{Real,Nothing}, alpha::Real)
        isnothing(bandwidth) ||
            bandwidth > 0 ||
            throw(ArgumentError("bandwidth has to be > 0"))
        new(bandwidth, alpha)
    end
end
DiffusionMaps(; bandwidth::Union{Real,Nothing}=nothing, alpha::Real=0.5) = DiffusionMaps(
    bandwidth, alpha
)

"""
    compute_embedding(distances, alg::DiffusionMaps; kwargs...) -> EmbeddingResult

When using the [`DiffusionMaps`](@ref) algorithm, the `res.info` dictionary contains

  - `res.info["elapsed"]`: the elapsed time
  - `res.info["bandwidth"]`: the used bandwidth
  - `res.info["eigvals"]`: the eigenvalues of the diffusion matrix
  - `res.info["eigvecs"]`: the eigenvectors of the diffusion matrix
  - `res.info["dimension_estimate"]`: estimate of the manifold dimension
"""
function compute_embedding(
    distances::AbstractMatrix{<:Real},
    alg::DiffusionMaps;
    n_coordinates::Int=typemax(Int),
    progress::Bool=false,
)::EmbeddingResult
    elapsed = @elapsed begin
        D_squared = distances .^ 2
        if isnothing(alg.bandwidth)
            ε, dimension_estimate = compute_optimal_bandwidth(D_squared)
        else
            ε = alg.bandwidth^2
            dimension_estimate = 2 * elasticity(ε, D_squared)
        end
        L = exp.(-D_squared ./ ε)

        vals, vecs = compute_diffusion_maps_from_similarity_matrix(L, alg.alpha)
        if n_coordinates < typemax(Int)
            n_coordinates += 1  # skip the first eigenvector
        end
        coordinates = (vecs .* vals')[:, 2:min(end, n_coordinates)]
    end

    info = Dict(
        "bandwidth" => isnothing(alg.bandwidth) ? sqrt(ε) : alg.bandwidth,
        "eigvals" => vals,
        "eigvecs" => vecs,
        "elapsed" => elapsed,
        "dimension_estimate" => dimension_estimate,
    )
    return EmbeddingResult(coordinates, info)
end

"""
    compute_diffusion_matrix(L::AbstractMatrix, α::Real) -> (M, D_inv)

Perform the diffusion map normalization on the similarity matrix `L` to construct
the symmetric matrix `M` for eigen-decomposition.
`D_inv` is a diagonal matrix ``D^{-1/2}`` which has to be used to transform the
eigenvectors of `M` back into the diffusion space, i.e.,
if ``v`` is an eigenvector of `M`, then `D_inv v` is the associated eigenvector
of the diffusion matrix.

Computing the spectrum of `M` instead of using the actual diffusion matrix
has the advantage that `M` is symmetric.
"""
function compute_diffusion_matrix(L::AbstractMatrix, α::Real)
    M = deepcopy(L)

    # D⁻ᵅ L D⁻ᵅ
    D_diag = vec(sum(M; dims=1))
    D_inv = Diagonal(1 ./ D_diag .^ α)
    rmul!(lmul!(D_inv, M), D_inv)

    # (Dᵅ)⁻¹ L
    # Generalized problem:
    # eigenvalues of (Dᵅ)⁻¹ L where L is symmetric: (Dᵅ)⁻¹ L v = lambda v
    # => L v = lambda Dᵅ v
    D = Diagonal(vec(sum(M; dims=1)))

    # Apply Laplacian normalization and transform into a symmetric problem
    D_inv = Diagonal(1 ./ D .^ (1 / 2))
    rmul!(lmul!(D_inv, M), D_inv)

    # symmetrize floating point errors
    @. M = (M + M') / 2
    return M, D_inv
end

function compute_diffusion_maps_from_similarity_matrix(L::AbstractMatrix, α::Real)
    M, D_inv = compute_diffusion_matrix(L, α)
    vals, vecs = eigen(Symmetric(M))

    # dominant spectrum first
    reverse!(vals)
    reverse!(vecs; dims=2)

    # transform into diffusion maps coordinates
    vecs = D_inv * vecs
    for v in eachcol(vecs)  # normalize
        v ./= norm(v)
    end
    return vals, vecs
end

"""
    compute_optimal_bandwidth(D_squared)

Given the distance matrix ``D`` (symmetric (n × n)-matrix with 0 on the diagonal),
compute the optimal ε for constructing the similarity matrix ``K`` with

```math
K_{ij} = exp(-D_{i,j}^2 / ε).
```

The input of this function is the squared distance matrix `D.^2`.
It finds the ε that maximizes the elasticity of the function

```math
S(ε) = 1/n^2 \\sum_{i,j} K_{ij}(ε)).
```

Returns a tuple `(ε, d)` where `d` is an estimate of the manifold dimension.
"""
function compute_optimal_bandwidth(D_squared::AbstractMatrix{<:Real})
    n = size(D_squared, 1)
    S(ϵ) = compute_means(D_squared, ϵ)[1]

    # We know that for small ϵ it is lim_{\epsilon -> 0}S(ϵ) = 1/n,
    # and for large ϵ it is lim_{\epsilon -> Inf}S(ϵ) = 1.
    # The relevant range of epsilons is in between.
    S_lower = (1.0 / n) + 0.001
    S_upper = 0.999
    stepfac = 2.0
    lower_bound = 1e-10
    while S(lower_bound) <= S_lower
        lower_bound *= stepfac
    end
    lower_bound /= stepfac

    upper_bound = 1e10
    while S(upper_bound) >= S_upper
        upper_bound /= stepfac
    end
    upper_bound *= stepfac

    # We find the optimal ϵ by simply sampling over the relevant range.
    # (The precision of a dedicated optimization algorithm is not needed.)
    # We sample 100 epsilons per order of magnitude in the relevant range:
    n_eps = log10(upper_bound / lower_bound) * 100.0 |> round |> Int
    n_eps = min(n_eps, 1000)
    epsilons = logrange(lower_bound, upper_bound, n_eps)
    elasts = Vector{Float64}(undef, length(epsilons))
    Threads.@threads for i in eachindex(epsilons)
        elasts[i] = elasticity(epsilons[i], D_squared)
    end
    optim_idx = argmax(elasts)
    optim_epsilon = epsilons[optim_idx]
    dimension_estimate = 2 * elasts[optim_idx]
    return (optim_epsilon, dimension_estimate)
end

# helper function computes the two sums in a single pass over D_squared:
# `sum_exp = mean(exp.(-D_squared ./ ϵ)) = S(ϵ)`
# `sum_d2_exp = mean(exp.(-D_squared ./ ϵ) .* D_squared) = dS(ϵ) * ϵ^2`
# The elasticity is then `ϵ * dS(ϵ) / S(ϵ) = sum_d2_exp / sum_exp / ϵ`
function compute_means(D_squared::Matrix{<:AbstractFloat}, ϵ::AbstractFloat)
    n = size(D_squared, 1)
    sum_exp = Float64(n) # contribution from the n diagonal elements
    sum_d2_exp = 0.0

    for j in 1:n
        for i in 1:(j - 1)
            d = D_squared[i, j]
            exp_val = exp(-d / ϵ)
            sum_exp += 2 * exp_val # 2 accounts for second triangular matrix
            sum_d2_exp += 2 * d * exp_val
        end
    end

    return sum_exp / n^2, sum_d2_exp / n^2
end

function elasticity(ϵ::AbstractFloat, D_squared::Matrix{<:AbstractFloat})
    sum_exp, sum_d2_exp = compute_means(D_squared, ϵ)
    return sum_d2_exp / sum_exp / ϵ
end
