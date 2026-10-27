@testset "compute_kernel_matrix" begin
    alg = GaussianDStatMMD(sqrt(2))
    seed!(1234)
    x = [rand(2, 4), rand(2, 3), rand(2, 2)]

    kmat = TransitionManifolds.compute_kernel_matrix(x, alg)
    @test size(kmat) == (3, 3)
    @test kmat[1, 1] == TransitionManifolds.kernel_eval(x[1], alg)
    @test kmat[2, 2] == TransitionManifolds.kernel_eval(x[2], alg)
    @test kmat[3, 3] == TransitionManifolds.kernel_eval(x[3], alg)
    @test kmat[1, 2] == TransitionManifolds.kernel_eval(x[1], x[2], alg)
    @test kmat[1, 3] == TransitionManifolds.kernel_eval(x[1], x[3], alg)
    @test kmat[2, 3] == TransitionManifolds.kernel_eval(x[2], x[3], alg)
end

@testset "convert_kernel_to_distance_matrix" begin
    # Dij = Kii + Kjj - 2Kij.
    # K is assmumed to be symmetric and only the upper triangular is used
    K = [3.0 2 1; 0 4 2; 0 0 5]
    D = [0 3.0 6; 3 0 5; 6 5 0]

    TransitionManifolds.convert_kernel_to_distance_matrix!(K)
    @test K == D
end

@testset "apply_ustat_correction!" begin
    @testset "scalar n_samples" begin
        K = diagm([1.0, 2.0, 3.0])
        TransitionManifolds.apply_ustat_correction!(K, 5, 1.0)
        @test K == diagm([1.0, 9 / 4, 14 / 4])
    end

    @testset "vector n_samples" begin
        K = diagm([1.0, 2.0, 3.0])
        TransitionManifolds.apply_ustat_correction!(K, [5, 3, 10], 1.0)
        @test K[1, 1] ≈ (5 * 1.0 - 1) / 4
        @test K[2, 2] ≈ (3 * 2.0 - 1) / 2
        @test K[3, 3] ≈ (10 * 3.0 - 1) / 9
    end

    @testset "off-diagonal is untouched" begin
        K = [3.0 2 1; 0 4 2; 0 0 5]
        expected_off_diag = copy(K)
        TransitionManifolds.apply_ustat_correction!(K, 5, 1.0)
        @test K[1, 2] == expected_off_diag[1, 2]
        @test K[1, 3] == expected_off_diag[1, 3]
        @test K[2, 3] == expected_off_diag[2, 3]
    end

    @testset "errors" begin
        K = diagm([1.0, 2.0])
        @test_throws ArgumentError TransitionManifolds.apply_ustat_correction!(K, 1, 1.0)
        @test_throws ArgumentError TransitionManifolds.apply_ustat_correction!(K, [1, 1], 1.0)
        @test_throws ArgumentError TransitionManifolds.apply_ustat_correction!(K, [5, 5, 5], 1.0)
    end
end
