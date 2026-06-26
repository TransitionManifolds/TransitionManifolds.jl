module TransitionManifolds

# ------------- Imports -----------------
using LinearAlgebra
using StatsBase: quantile, sample, median, cov, mean
using Distances: pairwise, pairwise!, SqEuclidean, Metric, Euclidean
using LoopVectorization: @turbo
using Tullio: @tullio
using ProgressMeter: Progress, next!

# ------------- include files -----------------
include("interface.jl")
export AbstractDataLayout, Contiguous, Jagged, layout
export TransitionDistanceProblem, n_anchors, n_samples, dimension
export cat_anchors, append_anchors!
export AbstractTransitionDistanceAlgorithm, TransitionDistanceResult, compute_distances
export AbstractEmbeddingAlgorithm, EmbeddingResult, compute_embedding
export compute_transition_manifold
export PreprocessResult, preprocess

include("utils.jl")
export normalize_cloud

include("td_gaussian_mmd.jl")
export GaussianVStatMMD, GaussianDStatMMD

include("td_kernel_mmd.jl")
export KernelVStatMMD, KernelDStatMMD

include("em_diffusion_maps.jl")
export DiffusionMaps

include("pre_trajs.jl")
export Trajectories, is_endpoint
export sample_points, farthest_point_sampling, FarthestPointSamplingResult

end # module TransitionManifolds
