using KernelFunctions

@testset "KernelVStatMMD" begin
    #NOTE: The code assumes a symetric kernel throughtout.
    @testset "Non-Symmetric kernels" begin
        kernel = ExponentialKernel(; metric=Distances.KLDivergence()) ∘ ScaleTransform(1.0)
        alg = KernelVStatMMD(kernel)
        x = rand(Float64, 2, 4, 3)
        @test_warn "The metric is not symmetric." compute_distances(x, alg)
    end
    @testset "compute_distances" begin
        # define Gaussian kernel
        bandwidth = 0.3
        kernel = ExponentialKernel(; metric=SqEuclidean()) ∘ ScaleTransform(1 / bandwidth)

        @testset "output" begin
            alg = KernelVStatMMD(kernel)
            x = rand(Float64, 2, 4, 3)
            sol = compute_distances(x, alg)

            @test size(sol.distances) == (3, 3)
            @test issymmetric(sol.distances)
            @test all(diag(sol.distances) .== 0)

            @test sol.info["elapsed"] > 0
        end

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

        # TODO: More tests. Compare to GaussianVStatMMD
    end
end

@testset "KernelDStatMMD" begin
    #NOTE: The code assumes a symetric kernel throughout.
    @testset "Non-Symmetric kernels" begin
        kernel = ExponentialKernel(; metric=Distances.KLDivergence()) ∘ ScaleTransform(1.0)
        alg = KernelDStatMMD(kernel)
        x = rand(Float64, 2, 4, 3)
        @test_warn "The metric is not symmetric." compute_distances(x, alg)
    end
    @testset "compute_distances" begin
        # define Gaussian kernel
        bandwidth = 0.3
        kernel = ExponentialKernel(; metric=SqEuclidean()) ∘ ScaleTransform(1 / bandwidth)

        @testset "output" begin
            alg = KernelDStatMMD(kernel)
            x = rand(Float64, 2, 4, 3)
            sol = compute_distances(x, alg)

            @test size(sol.distances) == (3, 3)
            @test issymmetric(sol.distances)
            @test all(diag(sol.distances) .== 0)

            @test sol.info["elapsed"] > 0
        end

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

        @testset "compare to GaussianDStatMMD" begin
            alg1 = KernelDStatMMD(kernel)
            alg2 = GaussianDStatMMD(bandwidth)
            seed!(1234)
            x = rand(3, 100, 10)

            D1 = compute_distances(x, alg1).distances
            D2 = compute_distances(x, alg2).distances
            @test D1 ≈ D2
        end

        # TODO: More tests.
    end
end
