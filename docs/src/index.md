# TransitionManifolds.jl


## Compute Distances

```@docs
TransitionDistanceProblem
```

```@docs
AbstractDataLayout
```

```@docs
Contiguous
```

```@docs
Jagged
```

```@docs
layout
```

```@docs
n_anchors
```

```@docs
n_samples
```

```@docs
dimension
```

```@docs
compute_distances
```

```@docs
TransitionDistanceResult
```

### Distance Algorithms

```@docs
AbstractTransitionDistanceAlgorithm
```

```@docs
GaussianDStatMMD
```

```@docs
GaussianVStatMMD
```

```@docs
GaussianUStatMMD
```

```@docs
KernelDStatMMD
```

```@docs
KernelVStatMMD
```

```@docs
KernelUStatMMD
```


## Compute Embedding

```@docs
compute_embedding
```

```@docs
EmbeddingResult
```

### Embedding Algorithms

```@docs
AbstractEmbeddingAlgorithm
```

```@docs
DiffusionMaps
```


## Preprocessing

```@docs
preprocess
```

```@docs
PreprocessResult
```

### Trajectories

```@docs
Trajectories
```

```@docs
is_endpoint
```

```@docs
sample_points
```

```@docs
farthest_point_sampling
```

```@docs
FarthestPointSamplingResult
```


## Helper functions

```@docs
compute_transition_manifold
```

```@docs
cat_anchors
```

```@docs
append_anchors!
```

```@docs
normalize_cloud
```


## GPU

The distances computation with the [`GaussianVStatMMD`](@ref) algorithm can be executed on Nvidia GPUs using `CUDA.jl`.
(Only for [`Contiguous`](@ref) data layout.)
Simply convert the input data to `CuArray`:
```julia
using CUDA

x = rand(2, 5, 20)
alg = GaussianVStatMMD()
res = compute_distances(cu(x), alg)
```
The `cu` function can also be applied directly to a [`TransitionDistanceProblem`](@ref) or a [`PreprocessResult`](@ref) to move the contained data to the GPU.
