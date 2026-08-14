module BenchmarkTrajs

using BenchmarkTools
using TransitionManifolds
using Random: seed!
using Distances: SqEuclidean

const SUITE = BenchmarkGroup()

seed!(123)

dist = SqEuclidean()
n_anchors = 500
trajs_small = Trajectories([rand(10, 40_000), rand(10, 60_000)])

SUITE["FarthestPointSampling"] = BenchmarkGroup()
SUITE["FarthestPointSampling"]["small"] = @benchmarkable farthest_point_sampling(
    $trajs_small, $n_anchors; dist=$dist
)

# trajs_big = Trajectories([rand(100, 80_000), rand(100, 120_000)])
end
