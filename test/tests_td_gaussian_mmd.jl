@testset "GaussianDStatMMD" begin
    @testset "kernel eval" begin
        alg = GaussianDStatMMD(sqrt(2))
        x = transpose([1.0 1; 2 3; 0 -1])  # n = 3, d = 2
        y = transpose([1.0 1; 2 0; 1 -1])
        x32 = Float32.(x)
        y32 = Float32.(y)

        @testset "xy Float64" begin
            expected = sum([1.0, exp(-4.5), exp(-0.5)]) / 3.0
            @test TransitionManifolds.kernel_eval(x, y, alg) ≈ expected
        end

        @testset "xy Float32" begin
            expected = sum([1.0, exp(-4.5), exp(-0.5)]) / 3.0
            @test TransitionManifolds.kernel_eval(x32, y32, alg) ≈ expected
        end

        @testset "x Float64" begin
            expected = sum([exp(-2.5), exp(-10)]) / 2.0
            @test TransitionManifolds.kernel_eval(x, alg) ≈ expected
        end

        @testset "x Float32" begin
            expected = sum([exp(-2.5), exp(-10)]) / 2.0
            @test TransitionManifolds.kernel_eval(x32, alg) ≈ expected
        end
    end

    @testset "kernel matrix" begin
        alg = GaussianDStatMMD(sqrt(2))
        seed!(1234)
        x = rand(2, 4, 3)

        kmat = TransitionManifolds.compute_kernel_matrix(x, alg)
        @test size(kmat) == (3, 3)
        @test kmat[1, 1] == TransitionManifolds.kernel_eval(x[:, :, 1], alg)
        @test kmat[2, 2] == TransitionManifolds.kernel_eval(x[:, :, 2], alg)
        @test kmat[3, 3] == TransitionManifolds.kernel_eval(x[:, :, 3], alg)
        @test kmat[1, 2] == TransitionManifolds.kernel_eval(x[:, :, 1], x[:, :, 2], alg)
        @test kmat[1, 3] == TransitionManifolds.kernel_eval(x[:, :, 1], x[:, :, 3], alg)
        @test kmat[2, 3] == TransitionManifolds.kernel_eval(x[:, :, 2], x[:, :, 3], alg)
    end

    @testset "constructor" begin
        @test_throws ArgumentError GaussianDStatMMD(0)
        @test_throws ArgumentError GaussianDStatMMD(-0.2)
    end

    @testset "compute_distances" begin
        @testset "output" begin
            alg = GaussianDStatMMD(0.123)
            x = rand(Float64, 2, 4, 3)
            x_j = [rand(2, 4), rand(2, 3), rand(2, 2)]
            prob = TransitionDistanceProblem(x)
            prob_j = TransitionDistanceProblem(x_j)

            @testset "$(layout(p))" for p in [prob, prob_j]
                res = compute_distances(p, alg)

                @test size(res.distances) == (3, 3)
                @test issymmetric(res.distances)
                @test all(diag(res.distances) .== 0)

                @test res.info["bandwidth"] == 0.123
                @test res.info["elapsed"] > 0
            end
        end

        @testset "types" begin
            alg = GaussianDStatMMD(1)
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
            alg = GaussianDStatMMD(1)
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
            alg = GaussianDStatMMD(0.3)
            seed!(1234)
            x = rand(2, 2000, 2)

            dmat = compute_distances(x, alg).distances
            @test dmat[1, 1] == 0
            @test dmat[2, 2] == 0
            @test dmat[1, 2] == dmat[2, 1]
            @test dmat[1, 2] < 0.02
        end

        @testset "automatic bandwidth" begin
            @testset "Contiguous" begin
                alg = GaussianDStatMMD()
                x = rand(Float64, 2, 100, 3)
                res = compute_distances(x, alg)
                @test res.info["bandwidth"] > 0
            end

            @testset "Jagged" begin
                alg = GaussianDStatMMD()
                x = [rand(2, 100), rand(2, 200), rand(2, 50)]
                res = compute_distances(x, alg)
                @test res.info["bandwidth"] > 0
            end
        end
    end
end
