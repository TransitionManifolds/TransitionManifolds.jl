# TransitionManifolds.jl

TransitionManifolds.jl is a Julia package for computing [transition manifolds](https://link.springer.com/article/10.1007/s00332-020-09668-z) from data.
The package provides functionality to:
1. **Preprocess data** into a suitable format
2. **Compute transition distances** between states using various methods  
3. **Compute an embedding** of the transition manifold for visualization and analysis

### Basic Workflow

How to use the package in a typical workflow:

```julia
using TransitionManifolds

# Step 1: Preprocess trajectory data
trajs = Trajectories(rand(3, 1000))  # Example trajectory data
pres = preprocess(trajs; anchors=50)  # Generate 50 anchors

# Step 2: Compute transition distances
distance_alg = GaussianDStatMMD()  # Choose a distance algorithm
dres = compute_distances(pres, distance_alg)

# Step 3: Compute embedding
embedding_alg = DiffusionMaps()
eres = compute_embedding(dres, embedding_alg; n_coordinates=2)
println("Coordinates: $(eres.coordinates)")
```


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

The distances computation with [`GaussianVStatMMD`](@ref) and [`GaussianUStatMMD`](@ref) algorithm can be executed on Nvidia GPUs using `CUDA.jl`.
(Only for [`Contiguous`](@ref) data layout.)
Simply convert the input data to `CuArray`:
```julia
using CUDA

x = rand(2, 5, 20)
alg = GaussianVStatMMD()
res = compute_distances(cu(x), alg)
```
The `cu` function can also be applied directly to a [`TransitionDistanceProblem`](@ref) or a [`PreprocessResult`](@ref) to move the contained data to the GPU.
