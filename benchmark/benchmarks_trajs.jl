module BenchmarkTrajs

using BenchmarkTools
using TransitionManifolds
using Random: seed!
using Distances: SqEuclidean

const SUITE = BenchmarkGroup()

seed!(123)

dist = SqEuclidean()
n_anchors = 1000
trajs_small = Trajectories([rand(10, 40_000), rand(10, 60_000)])
trajs_big = Trajectories([rand(100, 160_000), rand(100, 240_000)])

SUITE["FarthestPointSampling"] = BenchmarkGroup()
SUITE["FarthestPointSampling"]["small"] = @benchmarkable farthest_point_sampling(
    $trajs_small, $n_anchors; dist=$dist
)
SUITE["FarthestPointSampling"]["big"] = @benchmarkable farthest_point_sampling(
    $trajs_big, $n_anchors; dist=$dist
)

end
