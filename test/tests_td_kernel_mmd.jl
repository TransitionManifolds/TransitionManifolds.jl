using KernelFunctions

# a kernel with k(x,x) = kappa(0) = 1.5 != 1, used to test the `KernelUStatMMD`
# self-normalization check.
struct _OffsetTestKernel{M} <: KernelFunctions.SimpleKernel
    metric::M
end
KernelFunctions.kappa(::_OffsetTestKernel, d::Real) = exp(-d) + 0.5
KernelFunctions.metric(k::_OffsetTestKernel) = k.metric

#-----------------------------------------------------------------------------
# Shared behavior across KernelDStatMMD / KernelVStatMMD / KernelUStatMMD
#-----------------------------------------------------------------------------
@testset "Kernel*StatMMD shared behavior" begin
    # define Gaussian kernel
    bandwidth = 0.3
    kernel = ExponentialKernel(; metric=SqEuclidean()) ∘ ScaleTransform(1 / bandwidth)
    algs = [KernelDStatMMD(kernel), KernelVStatMMD(kernel), KernelUStatMMD(kernel)]

    @testset "output" begin
        @testset "$(nameof(typeof(alg)))" for alg in algs
            x = rand(Float64, 2, 4, 3)
            x_j = [rand(2, 4), rand(2, 3), rand(2, 2)]
            prob = TransitionDistanceProblem(x)
            prob_j = TransitionDistanceProblem(x_j)

            @testset "$(layout(p))" for p in [prob, prob_j]
                res = compute_distances(p, alg)

                @test size(res.distances) == (3, 3)
                @test issymmetric(res.distances)
                @test all(diag(res.distances) .== 0)

                @test res.info["elapsed"] > 0
            end
        end
    end

    @testset "types" begin
        types = [Float64, Float32, Float16]

        @testset "$(nameof(typeof(alg)))" for alg in algs
            @testset "$t" for t in types
                @testset "Contiguous" begin
                    x = rand(t, 2, 4, 3)
                    @test typeof(compute_distances(x, alg).distances) == Array{t,2}
                end

                @testset "Jagged" begin
                    x = [rand(t, 2, 4), rand(t, 2, 3), rand(t, 2, 2)]
                    @test typeof(compute_distances(x, alg).distances) == Array{t,2}
                end
            end
        end
    end

    @testset "cast" begin
        types = [Int64, Int32, UInt32]

        @testset "$(nameof(typeof(alg)))" for alg in algs
            @testset "$t" for t in types
                @testset "Contiguous" begin
                    x = rand(t, 2, 4, 3)
                    res = @test_logs (:info, r"Casting data") compute_distances(x, alg)
                    @test typeof(res.distances) == Array{Float32,2}
                end

                @testset "Jagged" begin
                    x = [rand(t, 2, 4), rand(t, 2, 3), rand(t, 2, 2)]
                    res = @test_logs (:info, r"Casting data") compute_distances(x, alg)
                    @test typeof(res.distances) == Array{Float32,2}
                end
            end
        end
    end

    @testset "compare to Gaussian*StatMMD" begin
        pairs = [
            (KernelDStatMMD(kernel), GaussianDStatMMD(bandwidth)),
            (KernelVStatMMD(kernel), GaussianVStatMMD(; bandwidth=bandwidth)),
            (KernelUStatMMD(kernel), GaussianUStatMMD(; bandwidth=bandwidth)),
        ]

        @testset "$(nameof(typeof(alg1)))" for (alg1, alg2) in pairs
            seed!(1234)
            x = rand(3, 100, 10)

            D1 = compute_distances(x, alg1).distances
            D2 = compute_distances(x, alg2).distances
            @test D1 ≈ D2
        end
    end
end

#-----------------------------------------------------------------------------
# KernelDStatMMD
#-----------------------------------------------------------------------------
@testset "KernelDStatMMD" begin
    @testset "warn if non-symmetric kernel" begin
        kernel = ExponentialKernel(; metric=Distances.KLDivergence()) ∘ ScaleTransform(1.0)
        alg = KernelDStatMMD(kernel)
        x = rand(Float64, 2, 4, 3)
        @test_warn "The metric is not symmetric." compute_distances(x, alg)
    end

    @testset "compute_distances" begin
        # define Gaussian kernel
        bandwidth = 0.3
        kernel = ExponentialKernel(; metric=SqEuclidean()) ∘ ScaleTransform(1 / bandwidth)

        @testset "convergence to 0" begin
            alg = KernelDStatMMD(kernel)
            seed!(1234)
            x = rand(2, 2000, 2)

            dmat = compute_distances(x, alg).distances
            @test dmat[1, 1] == 0
            @test dmat[2, 2] == 0
            @test dmat[1, 2] == dmat[2, 1]
            @test dmat[1, 2] < 0.02
        end
    end
end

#-----------------------------------------------------------------------------
# KernelVStatMMD
#-----------------------------------------------------------------------------
@testset "KernelVStatMMD" begin
    @testset "warn if non-symmetric kernel" begin
        kernel = ExponentialKernel(; metric=Distances.KLDivergence()) ∘ ScaleTransform(1.0)
        alg = KernelVStatMMD(kernel)
        x = rand(Float64, 2, 4, 3)
        @test_warn "The metric is not symmetric." compute_distances(x, alg)
    end

    @testset "compute_distances" begin
        # define Gaussian kernel
        bandwidth = 0.3
        kernel = ExponentialKernel(; metric=SqEuclidean()) ∘ ScaleTransform(1 / bandwidth)

        @testset "convergence to 0" begin
            alg = KernelVStatMMD(kernel)
            seed!(1234)
            x = rand(2, 2000, 2)

            dmat = compute_distances(x, alg).distances
            @test dmat[1, 1] == 0
            @test dmat[2, 2] == 0
            @test dmat[1, 2] == dmat[2, 1]
            @test dmat[1, 2] < 0.01
        end
    end
end

#-----------------------------------------------------------------------------
# KernelUStatMMD
#-----------------------------------------------------------------------------
@testset "KernelUStatMMD" begin
    @testset "errors if non-symmetric metric" begin
        kernel = ExponentialKernel(; metric=Distances.KLDivergence()) ∘ ScaleTransform(1.0)
        alg = KernelUStatMMD(kernel)
        x = rand(Float64, 2, 4, 3)
        @test_throws ArgumentError compute_distances(x, alg)
    end

    @testset "errors if kernel is not self-normalized" begin
        kernel = _OffsetTestKernel(SqEuclidean()) ∘ ScaleTransform(1.0)
        alg = KernelUStatMMD(kernel)
        x = rand(Float64, 2, 4, 3)
        @test_throws ArgumentError compute_distances(x, alg)
    end

    @testset "compute_distances" begin
        # define Gaussian kernel
        bandwidth = 0.3
        kernel = ExponentialKernel(; metric=SqEuclidean()) ∘ ScaleTransform(1 / bandwidth)

        @testset "weights not supported" begin
            alg = KernelUStatMMD(kernel)
            x = rand(2, 4, 3)
            w = rand(4, 3)
            prob = TransitionDistanceProblem(x, w)
            @test_throws ErrorException compute_distances(prob, alg)
        end
    end
end
