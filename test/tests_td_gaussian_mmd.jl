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

@testset "GaussianVStatMMD" begin
    @testset "kernel_eval" begin
        alg = GaussianVStatMMD(; bandwidth=sqrt(2))
        x = transpose([1.0 1; 2 3; 0 -1])
        y = transpose([1.0 1; 2 0])

        @testset "not weighted" begin
            expected = sum([1, exp(-1), exp(-2.5), exp(-4.5), exp(-2.5), exp(-2.5)]) / 6
            @test TransitionManifolds.kernel_eval(x, y, alg) ≈ expected
        end

        @testset "weighted" begin
            wx = [0.5, 0.3, 0.2]
            wy = [0.1, 0.9]
            expected = sum([
                0.5 * 0.1 * 1,
                0.5 * 0.9 * exp(-1),
                0.3 * 0.1 * exp(-2.5),
                0.3 * 0.9 * exp(-4.5),
                0.2 * 0.1 * exp(-2.5),
                0.2 * 0.9 * exp(-2.5),
            ])
            @test TransitionManifolds.kernel_eval(x, y, wx, wy, alg) ≈ expected
        end
    end

    @testset "compute_distances" begin
        @testset "output" begin
            alg = GaussianVStatMMD(; bandwidth=0.123)
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

        @testset "blocksize" begin
            # test same result for different blocksizes
            alg = GaussianVStatMMD(; bandwidth=0.123)
            x = rand(Float64, 2, 5, 20)
            prob = TransitionDistanceProblem(x)
            blocksizes = [2, 3, 19, 20, 50]

            alg = GaussianVStatMMD(; bandwidth=0.123, blocksize=1)
            expected = compute_distances(prob, alg).distances

            @testset "blocksize $b" for b in blocksizes
                alg = GaussianVStatMMD(; bandwidth=0.123, blocksize=b)
                res = compute_distances(prob, alg)
                @test res.distances ≈ expected
            end
        end

        @testset "types" begin
            alg = GaussianVStatMMD(; bandwidth=1, blocksize=1)
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
            alg = GaussianVStatMMD(; bandwidth=1, blocksize=1)
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
            alg = GaussianVStatMMD(; bandwidth=0.3)
            seed!(1234)
            x = rand(2, 2000, 2)

            dmat = compute_distances(x, alg).distances
            @test dmat[1, 1] == 0
            @test dmat[2, 2] == 0
            @test dmat[1, 2] == dmat[2, 1]
            @test dmat[1, 2] < 0.001
        end

        @testset "automatic bandwidth" begin
            @testset "Contiguous" begin
                alg = GaussianVStatMMD()
                x = rand(Float64, 2, 100, 3)
                res = compute_distances(x, alg)
                @test res.info["bandwidth"] > 0
            end

            @testset "Jagged" begin
                alg = GaussianVStatMMD()
                x = [rand(2, 100), rand(2, 200), rand(2, 50)]
                res = compute_distances(x, alg)
                @test res.info["bandwidth"] > 0
            end
        end
    end

    @testset "compute_distances (weighted)" begin
        @testset "output" begin
            alg = GaussianVStatMMD(; bandwidth=0.123, blocksize=1)
            x = rand(2, 4, 3)
            w = rand(4, 3)
            x_j = [rand(2, 4), rand(2, 3), rand(2, 2)]
            w_j = [rand(4), rand(3), rand(2)]
            prob = TransitionDistanceProblem(x, w)
            prob_j = TransitionDistanceProblem(x_j, w_j)

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
            alg = GaussianVStatMMD(; bandwidth=1, blocksize=1)
            types = [Float64, Float32, Float16]

            @testset "$t" for t in types
                @testset "Contiguous" begin
                    x = rand(t, 2, 4, 3)
                    w = rand(t, 4, 3)
                    @test typeof(compute_distances(x, alg; weights=w).distances) ==
                        Array{t,2}
                end

                @testset "Jagged" begin
                    x = [rand(t, 2, 4), rand(t, 2, 3), rand(t, 2, 2)]
                    w = [rand(t, 4), rand(t, 3), rand(t, 2)]
                    @test typeof(compute_distances(x, alg; weights=w).distances) ==
                        Array{t,2}
                end
            end
        end

        @testset "cast" begin
            alg = GaussianVStatMMD(; bandwidth=1, blocksize=1)
            types = [Int64, Int32, UInt32]

            @testset "$t" for t in types
                @testset "Contiguous" begin
                    x = rand(t, 2, 4, 3)
                    w = rand(t, 4, 3)
                    res = @test_logs (:info, r"Casting data") compute_distances(
                        x, alg; weights=w
                    )
                    @test typeof(res.distances) == Array{Float32,2}
                end

                @testset "Jagged" begin
                    x = [rand(t, 2, 4), rand(t, 2, 3), rand(t, 2, 2)]
                    w = [rand(t, 4), rand(t, 3), rand(t, 2)]
                    res = @test_logs (:info, r"Casting data") compute_distances(
                        x, alg; weights=w
                    )
                    @test typeof(res.distances) == Array{Float32,2}
                end
            end
        end

        @testset "compare to unweighted" begin
            alg = GaussianVStatMMD(; bandwidth=0.123)

            @testset "Contiguous" begin
                x = rand(2, 8, 4)
                w = zeros(8, 4)
                w[5:8, :] .= 1
                x_eff = x[:, 5:8, :]

                @test compute_distances(x, alg; weights=w).distances ≈
                    compute_distances(x_eff, alg).distances
            end

            @testset "Jagged" begin
                x = [rand(2, 8), rand(2, 8), rand(2, 8)]
                w = [zeros(8), zeros(8), ones(8)]
                w[1][5:8] .= 1
                w[2][1:2] .= 1
                x_eff = [x[1][:, 5:8], x[2][:, 1:2], x[3]]

                @test compute_distances(x, alg; weights=w).distances ≈
                    compute_distances(x_eff, alg).distances
            end
        end
    end
end
