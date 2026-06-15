# TransitionManifolds.jl

```@contents
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
KernelDStatMMD
```

```@docs
KernelVStatMMD
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

```@docs
Trajectories
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
