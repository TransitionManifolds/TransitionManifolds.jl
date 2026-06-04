module BenchmarkTD

using BenchmarkTools
using TransitionManifolds
using Random: seed!
using KernelFunctions
using Distances: SqEuclidean

const SUITE = BenchmarkGroup()

seed!(123)
d_big = 1000
x_big_64 = rand(d_big, 400, 400)
x_big_32 = Float32.(x_big_64)

d_medium = 500
x_medium_64 = rand(d_medium, 200, 400)
x_medium_32 = Float32.(x_medium_64)

d_small = 50
x_small_64 = rand(d_small, 200, 400)
x_small_32 = Float32.(x_small_64)

# --------- GaussianDStatMMD ---------
SUITE["GaussianDStatMMD"] = BenchmarkGroup()

alg = GaussianDStatMMD(sqrt(d_big / 2.0))
SUITE["GaussianDStatMMD"]["Big_Float64"] = @benchmarkable compute_distances($x_big_64, $alg)
SUITE["GaussianDStatMMD"]["Big_Float32"] = @benchmarkable compute_distances($x_big_32, $alg)

alg = GaussianDStatMMD(sqrt(d_medium / 2.0))
SUITE["GaussianDStatMMD"]["Medium_Float64"] = @benchmarkable compute_distances(
    $x_medium_64, $alg
)
SUITE["GaussianDStatMMD"]["Medium_Float32"] = @benchmarkable compute_distances(
    $x_medium_32, $alg
)

alg = GaussianDStatMMD(sqrt(d_small / 2.0))
SUITE["GaussianDStatMMD"]["Small_Float64"] = @benchmarkable compute_distances(
    $x_small_64, $alg
)
SUITE["GaussianDStatMMD"]["Small_Float32"] = @benchmarkable compute_distances(
    $x_small_32, $alg
)

# --------- KernelDStatMMD ---------
gaussian_kernel(bandwidth) =
    ExponentialKernel(; metric=SqEuclidean()) ∘ ScaleTransform(1 / bandwidth)
SUITE["KernelDStatMMD"] = BenchmarkGroup()

alg = KernelDStatMMD(gaussian_kernel(sqrt(d_big / 2.0)))
SUITE["KernelDStatMMD"]["Big_Float64"] = @benchmarkable compute_distances($x_big_64, $alg)
SUITE["KernelDStatMMD"]["Big_Float32"] = @benchmarkable compute_distances($x_big_32, $alg)

alg = KernelDStatMMD(gaussian_kernel(sqrt(d_medium / 2.0)))
SUITE["KernelDStatMMD"]["Medium_Float64"] = @benchmarkable compute_distances(
    $x_medium_64, $alg
)
SUITE["KernelDStatMMD"]["Medium_Float32"] = @benchmarkable compute_distances(
    $x_medium_32, $alg
)

alg = KernelDStatMMD(gaussian_kernel(sqrt(d_small / 2.0)))
SUITE["KernelDStatMMD"]["Small_Float64"] = @benchmarkable compute_distances(
    $x_small_64, $alg
)
SUITE["KernelDStatMMD"]["Small_Float32"] = @benchmarkable compute_distances(
    $x_small_32, $alg
)

end
