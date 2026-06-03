using KernelFunctions

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

        @testset "output" begin
            alg = KernelDStatMMD(kernel)
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

        @testset "types" begin
            alg = KernelDStatMMD(kernel)
            types = [Float64, Float32, Float16]

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

        @testset "cast" begin
            alg = KernelDStatMMD(kernel)
            types = [Int64, Int32, UInt32]

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
    end
end

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

        @testset "output" begin
            alg = KernelVStatMMD(kernel)

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

        @testset "types" begin
            alg = KernelVStatMMD(kernel)
            types = [Float64, Float32, Float16]

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

        @testset "cast" begin
            alg = KernelVStatMMD(kernel)
            types = [Int64, Int32, UInt32]

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

        # TODO: Compare to GaussianVStatMMD
    end
end
