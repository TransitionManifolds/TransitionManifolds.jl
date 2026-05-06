module TransitionManifolds

# ------------- Imports -----------------
using LinearAlgebra
using StatsBase: quantile, sample, median, cov
using Distances: pairwise, SqEuclidean
using LoopVectorization: @turbo
using Tullio: @tullio
using ProgressMeter: Progress, next!

# ------------- include files -----------------
include("interface.jl")
export AbstractTransitionDistanceAlgorithm, TransitionDistanceSolution, compute_distances
export AbstractEmbeddingAlgorithm, EmbeddingSolution, compute_embedding
export compute_transition_manifold

include("utils.jl")
export normalize_cloud

include("td_gaussian_mmd.jl")
export GaussianVStatMMD, GaussianDStatMMD

include("td_kernel_mmd.jl")
export KernelVStatMMD, KernelDStatMMD

include("em_diffusion_maps.jl")
export DiffusionMaps

end # module TransitionManifolds
