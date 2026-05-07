@testset "DiffusionMaps" begin
    @testset "compute_diffusion_maps_from_similarity_matrix" begin
        L = [1 0.7 0.3; 0.7 1 0.3; 0.3 0.3 1]
        alpha = 0.5

        # compute expected vals and vecs
        D_inv = Diagonal([1.0 / sqrt(2); 1.0 / sqrt(2); sqrt(5.0 / 8.0)])
        M = D_inv * L * D_inv
        # M = [0.5 0.35 0.167705; 0.35 0.5 0.167705; 0.167705 0.167705 0.625]
        # D_inv = Diagonal([1.0 / 1.017705; 1.0 / 1.017705; 1.0 / 0.96041])
        D_inv = inv(Diagonal([sum(M[1, :]), sum(M[2, :]), sum(M[3, :])]))
        M = D_inv * M
        F = eigen(M)
        sort_idx = sortperm(F.values; rev=true)
        expected_eigvals = F.values[sort_idx]
        expected_eigvecs = F.vectors[:, sort_idx]

        eigvals, eigvecs = TransitionManifolds.compute_diffusion_maps_from_similarity_matrix(
            L, alpha
        )

        @test eigvals ≈ expected_eigvals
        for (v1, v2) in zip(eachcol(eigvecs), eachcol(expected_eigvecs))
            @test v1 ≈ v2 || v1 ≈ -v2
        end
    end

    # construct valid distance matrix
    seed!(123)
    D = rand(10, 10)
    D = 0.5 * (D + D')
    D[diagind(D)] .= 0

    @testset "compute_optimal_bandwidth" begin
        (ε, d) = TransitionManifolds.compute_optimal_bandwidth(D)
        @test ε > 0
        @test d > 0
    end

    @testset "constructor" begin
        @test_throws ArgumentError DiffusionMaps(bandwidth=0)
        @test_throws ArgumentError DiffusionMaps(bandwidth=-0.2)
    end

    @testset "compute_embedding" begin
        @testset "output" begin
            alg = DiffusionMaps(0.123, 0.5)
            res = compute_embedding(D, alg; n_coordinates=3)
            @test size(res.coordinates) == (10, 3)

            @test res.info["elapsed"] > 0
            @test res.info["bandwidth"] == 0.123
            @test size(res.info["eigvals"]) == (10,)
            @test size(res.info["eigvecs"]) == (10, 10)
            @test res.info["dimension_estimate"] > 0

            @test res.coordinates[:, 1] ==
                res.info["eigvals"][2] * res.info["eigvecs"][:, 2]
        end

        @testset "automatic bandwidth" begin
            alg = DiffusionMaps()
            res = compute_embedding(D, alg)
            @test res.info["bandwidth"] > 0
        end
    end
end

# TODO: more tests for diffusion maps
# take some tests e.g. from
# from https://github.com/DiffusionMapsAcademics/pyDiffMap/blob/master/tests/test_diffusionmap.py
